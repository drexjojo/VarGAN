import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import *
from hyperparameters import *
from data_process import *
from utils import *
# from similarity.damerau import Damerau
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score


c_t = .1

# damerau = Damerau()

SAVE_FILE = "../data/trained_models/VarGAN.chkpt"

def calculate_metric(predictions,targets,index2word):
	LD_distance = 0
	targets = targets.squeeze(0)
	predictions = "".join([index2word[j] for j in predictions])
	targets = "".join([index2word[j] for j in targets.squeeze().cpu().numpy().tolist()][1:])
	LD_distance = damerau.distance(predictions,targets)/max(len(predictions),len(targets))
	LD_similarity = 1 - LD_distance

	return LD_distance, LD_similarity

def train_epoch(GAN, train_loader, opt_disc,opt_gen, device):
	epoch_loss_G = 0
	epoch_loss_D = 0
	epoch_disc_accuracy = 0
	disc_norm = 0
	epoch_gen_accuracy = 0
	gen_norm = 0
	acc_calc = ModifiedLoss().to(device)

	count = 0
	for batch in tqdm(train_loader, mininterval=2,desc='  - (Training)   ', leave=False):
		sequences, sequence_lengths, targets, target_lengths = batch
		sequences = sequences.to(device)
		sequence_lengths = sequence_lengths.to(device)
		targets = targets.to(device)
		target_lengths = target_lengths.to(device)

		mask = torch.gt(targets[:, 1:], 0).float()

		count +=1

		if count % TRAINING_RATIO != 0:
			GAN.generator.zero_grad()
			opt_gen.zero_grad()

			outputs,_,ael_responses = GAN.generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
			# Generator accuracy
			cont_outputs = outputs.contiguous().view(-1,outputs.shape[-1])
			cont_targets = targets[:,1:].contiguous().view(-1)
			acc_gen = acc_calc.get_accuracy(cont_outputs,cont_targets)
			epoch_gen_accuracy += acc_gen
			gen_norm += 1

			embedded_query = GAN.generator.embedding(sequences) #[batch_size,max_len,emb_dim]

			fake_responses = ael_responses * (mask.unsqueeze(-1).expand_as(ael_responses))
			D_fake,A_fake = GAN.discriminator(embedded_query, fake_responses) #[batch_size, 1]
			G_loss = -torch.mean(D_fake)
			epoch_loss_G += G_loss
			G_loss.backward()
			opt_gen.step()


		else:
			GAN.discriminator.zero_grad()
			opt_disc.zero_grad()

			real_responses = GAN.generator.embedding(targets[:,1:]).detach() #[batch_size,max_len,emb_dim]
			real_responses = real_responses * (mask.unsqueeze(-1).expand_as(real_responses))
			embedded_query = GAN.generator.embedding(sequences).detach() #[batch_size,max_len,emb_dim]
			D_real,_ = GAN.discriminator(embedded_query, real_responses) #[batch_size, 1]

			_,_,ael_responses = GAN.generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
			ael_responses = ael_responses.detach()
			fake_responses = ael_responses * (mask.unsqueeze(-1).expand_as(ael_responses))
			D_fake,_ = GAN.discriminator(embedded_query, fake_responses) #[batch_size, 1]

			predictions = torch.cat((torch.sigmoid(D_fake).squeeze(1),torch.sigmoid(D_real).squeeze(1))).to(device)
			real_values = torch.cat((torch.zeros(D_fake.shape[0]),torch.ones(D_real.shape[0]))).to(device)
			acc_disc = accuracy_score(real_values.detach().cpu().numpy(),torch.round(predictions).detach().cpu().numpy())
			epoch_disc_accuracy += acc_disc
			disc_norm += 1

			# print("\n Disc_accuracy : ",acc_disc)

			D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
			epoch_loss_D += D_loss
			D_loss.backward()
			opt_disc.step()

			for p in GAN.discriminator.parameters():
				p.data.clamp_(-c_t,c_t)

	
	epoch_loss_G /= gen_norm
	epoch_loss_D /= disc_norm
	epoch_gen_accuracy /= gen_norm
	epoch_disc_accuracy /= disc_norm

	return epoch_loss_D,epoch_loss_G,epoch_disc_accuracy,epoch_gen_accuracy

def eval_epoch(generator,valid_loader, device,index2word):
	generator.eval()
	avg_LD_sim  = 0
	avg_LD_dist = 0
	counter = 0
	with torch.no_grad():
		for batch in tqdm(valid_loader, mininterval=2,desc='  - (Validating)   ', leave=False):
			counter += 1
			sequences,sequence_lengths,targets, target_lengths = batch
			sequences = sequences.to(device)
			targets = targets.to(device)

			embedded_inputs = generator.embedding(sequences) #[batch,max_len,emb_dim]
			_, (hidden,cell)  = generator.encoder(embedded_inputs)
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
			cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)

			predictions = []
			outputs = []
			curr_state = hidden #[1,batch_size,hidden_size]
			curr_cell = cell
			next_word_embedding = generator.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim]

			while True:               
				output, (curr_state,curr_cell) = generator.decoder(next_word_embedding, (curr_state,curr_cell)) #output = [batch_size,1,hidden]
				output = generator.dense(output) # [batch_size, 1 , vocab_size]
				outputs.append(output)
				softmax_output = F.softmax(output.squeeze(1),dim=1)
				dec_inp_var = torch.max(softmax_output,dim=1,keepdim=True)[1] #[batch_size,1]
				next_word_embedding = torch.mm(softmax_output,generator.embedding.weight).unsqueeze(1)
				predictions.append(dec_inp_var.item())
				# next_word_embedding = generator.embedding(dec_inp_var)
				if dec_inp_var.item() == EOS_ID :
					break

			outputs = torch.cat(outputs,dim =1)
			LD_distance,LD_similarity = calculate_metric(predictions,targets,index2word)
			
			
			# print("-------------------------------------------")
			# print(counter,"/",len(valid_dset))
			# print("Input           : "," ".join([model_data.index2word[i] for i in sequences.squeeze().cpu().numpy().tolist()]))
			# print("Predicted       : "," ".join([model_data.index2word[i] for i in predictions]))
			# print("Target          : "," ".join([model_data.index2word[i] for i in targets.squeeze().cpu().numpy().tolist()][1:]))
			# print("LD_distance     : ",LD_distance)
			# print("LD_similarity   : ",LD_similarity)
			# print("-------------------------------------------")
			# print()
			avg_LD_dist += LD_distance
			avg_LD_sim += LD_similarity

	avg_LD_dist /= len(valid_loader)
	avg_LD_sim /=len(valid_loader)
	
	return avg_LD_dist, avg_LD_sim

def adversarial_training(GAN, train_loader, valid_loader, opt_disc,opt_gen, device):

	best_acc = 0
	for epoch_i in range(EPOCH):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_loss_D, train_loss_G,disc_accuracy,gen_accuracy = train_epoch(GAN, train_loader, opt_disc,opt_gen, device)
		
		print('  - (Training)     Loss_G: {loss_G: 8.5f} || Loss_D: {loss_D: 8.5f} || Time Taken : {elapse:3.3f} min || Disc_accuracy : {d_acc:8.5f} || Gen_accuracy : {g_acc:8.5f}'.format(
		loss_D=train_loss_D,loss_G=train_loss_G,d_acc=disc_accuracy,g_acc=gen_accuracy,elapse=(time.time()-start)/60))

		# start = time.time()
		# LD_distance,LD_similarity = eval_epoch(GAN.generator, valid_loader, device,index2word)
		# print('  - (Validating)   LD_similarity: {simi: 8.5f} <-> LD_distance: {dist: 8.5f} <-> Time Taken : {elapse:3.3f} min'.format(
		# simi=LD_similarity,dist=LD_distance,elapse=(time.time()-start)/60))
		
		# # # print(valid_loss_G)
		# # # print(best_valid)

		# if LD_similarity >= best_LD:
		# 	best_LD = LD_similarity
		# 	state_dict = GAN.state_dict()
		# 	checkpoint = {'state_dict': state_dict,
		# 					'epoch': epoch_i,
		# 					'LD_distance':LD_distance,
		# 					'LD_similarity':LD_similarity}
		# 	torch.save(checkpoint, SAVE_FILE)
		# 	print('[INFO] -> The checkpoint file has been updated.')

def inference():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)
	print("[INFO] -> Loading Preprocessed Data ...")
	model_data = torch.load("../data/model_data_BPI.pt")
	print("[INFO] -> Done!")

	generator = Encoder_Decoder(vocab_size=len(model_data.word2index))
	discriminator = Discriminator_CNN(128,eval("[1,2]"))
	GAN = GAN_AEL(generator,discriminator).to(device)
	GAN.load_state_dict(torch.load("../data/trained_models/GAN_model.chkpt")["state_dict"])
	# print("done")
	
	valid_dset = Driver_Data(
		data = model_data.test_data,
		targets = model_data.test_targets,
		word2index = model_data.word2index)

	valid_loader = DataLoader(valid_dset, batch_size = 1, shuffle = False, num_workers = 1,collate_fn=pack_collate_fn) #Load Validation data
	GAN.generator.eval()
	avg_LD_sim  = 0
	avg_LD_dist = 0
	counter = 0
	with torch.no_grad():
		for batch in valid_loader:
			counter +=1
			sequences,sequence_lengths,targets, target_lengths = batch
			sequences = sequences.to(device)
			targets = targets.to(device)

			embedded_inputs = GAN.generator.embedding(sequences) #[batch,max_len,emb_dim]
			_, (hidden,cell)  = GAN.generator.encoder(embedded_inputs)
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)
			cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1).unsqueeze(0)

			predictions = []
			outputs = []
			curr_state = hidden #[1,batch_size,hidden_size]
			curr_cell = cell
			next_word_embedding = GAN.generator.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim]

			while True:               
				output, (curr_state,curr_cell) = GAN.generator.decoder(next_word_embedding, (curr_state,curr_cell)) #output = [batch_size,1,hidden]
				output = GAN.generator.dense(output) # [batch_size, 1 , vocab_size]
				outputs.append(output)
				softmax_output = F.log_softmax(output.squeeze(1),dim=1)
				dec_inp_var = torch.max(softmax_output,dim=1,keepdim=True)[1] #[batch_size,1]
				predictions.append(dec_inp_var.item())
				next_word_embedding = torch.mm(softmax_output,generator.embedding.weight).unsqueeze(1)
				# next_word_embedding = GAN.generator.embedding(dec_inp_var)
				if dec_inp_var.item() == EOS_ID :
					break
				if len(outputs) > 233 :
					print("shizz")
					break

			outputs = torch.cat(outputs,dim =1)
			# print("done")
			LD_distance,LD_similarity = calculate_metric(predictions,targets,model_data.index2word)
			
			
			print("-------------------------------------------")
			print(counter,"/",len(valid_dset))
			print("Input           : "," ".join([model_data.index2word[i] for i in sequences.squeeze().cpu().numpy().tolist()]))
			print("Predicted       : "," ".join([model_data.index2word[i] for i in predictions]))
			print("Target          : "," ".join([model_data.index2word[i] for i in targets.squeeze().cpu().numpy().tolist()][1:]))
			print("LD_distance     : ",LD_distance)
			print("LD_similarity   : ",LD_similarity)
			print("-------------------------------------------")
			print()
			avg_LD_dist += LD_distance
			avg_LD_sim += LD_similarity
	print("avg_LD_dist : ",avg_LD_dist/len(valid_dset))
	print("avg_LD_sim  : ",avg_LD_sim/len(valid_dset))

def main():

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# device = torch.device("cpu")
	print("[INFO] -> Using Device : ",device)
	if device.type == 'cuda':
		print(torch.cuda.get_device_name(1))
		print('	Memory Usage : ')
		print('	Allocated    : ', round(torch.cuda.memory_allocated(1)/1024**3,1), 'GB')
		print('	Cached		 : ', round(torch.cuda.memory_cached(1)/1024**3,1), 'GB')

		
	
	print("[INFO] -> Loading Preprocessed Data ...")
	model_data = torch.load("../data/model_data_bpi17.pt",map_location='cpu')
	print("[INFO] -> Done!")

	generator = New_VarGenerator(vocab_size=len(model_data.word2index))
	generator.load_state_dict(torch.load("../data/trained_models/New_VarGenerator.chkpt",map_location='cpu' )["state_dict"])
	# generator = Encoder_Decoder(vocab_size=len(model_data.word2index))
	# generator.load_state_dict(torch.load("../data/trained_models/Encoder_Decoder.chkpt")["state_dict"])
	print("Generator : ")
	print(generator)
	discriminator = Discriminator_CNN(64,[1,2])
	# discriminator.load_state_dict(torch.load("../data/trained_models/discriminator_CNN.chkpt")["state_dict"])
	print()
	print("Discriminator : ")
	print(discriminator)


	GAN = GAN_AEL(generator,discriminator).to(device)
	

	opt_disc = optim.RMSprop(GAN.discriminator.parameters(),lr=0.00005)
	opt_gen  = optim.RMSprop(GAN.generator.decoder.parameters(),lr=0.0005)

	train_dset = Driver_Data(
		data    = model_data.train_data,
		targets = model_data.train_targets,
		word2index = model_data.word2index)

	train_loader = DataLoader(train_dset, batch_size = BATCH_SIZE,shuffle = True, num_workers = 10,collate_fn=pack_collate_fn) #Load training data

	valid_dset = Driver_Data(
		data = model_data.test_data,
		targets = model_data.test_targets,
		word2index = model_data.word2index)

	valid_loader = DataLoader(valid_dset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn) #Load Validation data

	adversarial_training(GAN, train_loader, valid_loader, opt_disc,opt_gen, device)

if __name__ == '__main__':
	main()
	# inference()