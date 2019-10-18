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
from utils import ModifiedLoss
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
from jellyfish import damerau_levenshtein_distance


SAVE_FILE = "../data/Helpdesk/trained_models/New_VarGAN.chkpt"
TASK = "suffix_prediction"

def train_epoch(generator, train_loader, optimizer, device, loss_func):
	generator.train()
	epoch_CE_loss = 0
	epoch_KL_loss = 0
	epoch_accuracy = 0
	
	for batch in tqdm(train_loader, mininterval=2,desc='  - (Training)   ', leave=False):

		sequences, sequence_lengths, targets, target_lengths = batch
		sequences = sequences.to(device)
		sequence_lengths = sequence_lengths.to(device)
		targets = targets.to(device)
		batch_size = sequences.shape[0]

		#For Enc-Dec with AEL
		outputs,ael_outputs = generator(sequences,sequence_lengths,targets,task=TASK)
		
		#For Conditional VAE with AEL
		# outputs,kld,ael_outputs = generator(sequences,sequence_lengths,targets,task=TASK)

		if TASK == "next_activity_prediction":
			loss,loss_dict,accuracy = loss_func.compute_batch_loss(outputs,targets[:,1],batch_size)

		elif TASK =="suffix_prediction":
			loss,loss_dict,accuracy = loss_func.compute_batch_loss(outputs,targets[:,1:],batch_size)
		

		epoch_KL_loss += loss_dict["KLDLoss"]
		epoch_CE_loss += loss_dict["CELoss"]
		epoch_accuracy += float(accuracy)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

	epoch_CE_loss  /= len(train_loader)
	epoch_KL_loss  /= len(train_loader)
	epoch_accuracy /= len(train_loader)

	return epoch_CE_loss,epoch_KL_loss,epoch_accuracy

def eval_epoch(generator, valid_loader, device, loss_func):
	generator.eval()
	epoch_CE_loss = 0
	epoch_KL_loss = 0
	epoch_accuracy = 0
	with torch.no_grad():
		for batch in tqdm(valid_loader, mininterval=2,desc='  - (Validating)   ', leave=False):

			sequences,sequence_lengths,targets, target_lengths = batch
			sequences = sequences.to(device)
			sequence_lengths = sequence_lengths.to(device)
			targets = targets.to(device)
			target_lengths = target_lengths.to(device)
			batch_size = sequences.shape[0]

			#For Enc-Dec with AEL
			outputs,ael_outputs = generator(sequences,sequence_lengths,targets,task=TASK)

			#For Conditional VAE with AEL
			# outputs,kld,ael_outputs = generator(sequences,sequence_lengths,targets,task=TASK)

			if TASK == "next_activity_prediction":
				loss,loss_dict,accuracy = loss_func.compute_batch_loss(outputs,targets[:,1],batch_size)

			elif TASK =="suffix_prediction":
				loss,loss_dict,accuracy = loss_func.compute_batch_loss(outputs,targets[:,1:],batch_size)

			epoch_KL_loss  += loss_dict["KLDLoss"]
			epoch_CE_loss  += loss_dict["CELoss"]
			epoch_accuracy += float(accuracy)
				
	epoch_CE_loss  /= len(valid_loader)
	epoch_KL_loss  /= len(valid_loader)
	epoch_accuracy /= len(valid_loader)
	
	return epoch_CE_loss,epoch_KL_loss,epoch_accuracy

def train_generator(generator, train_loader, valid_loader, optimizer, device, loss_func):

	best_valid_acc = 0
	for epoch_i in range(EPOCH):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_CE_loss,train_KL_loss,train_acc = train_epoch(generator, train_loader, optimizer, device, loss_func)
		print('  - (Training)       CE_Loss: {ce_loss: 8.5f} ||  KL_Loss: {kl_loss: 8.5f} || Accuracy: {acc:3.3f} || Time Taken: {elapse:3.3f} min'.format(
		ce_loss=train_CE_loss,kl_loss=train_KL_loss,acc=train_acc,elapse=(time.time()-start)/60))

		start = time.time()
		valid_CE_loss,valid_KL_loss,valid_acc = eval_epoch(generator, valid_loader, device, loss_func)
		print('  - (Validating)     CE_Loss: {ce_loss: 8.5f} ||  KL_Loss: {kl_loss: 8.5f} || Accuracy: {acc:3.3f} || Time Taken: {elapse:3.3f} min'.format(
		ce_loss=valid_CE_loss,kl_loss=valid_KL_loss,acc=valid_acc,elapse=(time.time()-start)/60))
		
		if valid_acc >= best_valid_acc:
			best_valid_acc = valid_acc
			generator_state_dict = generator.state_dict()
			checkpoint = {'state_dict': generator_state_dict,
							'epoch': epoch_i,
							'CE_loss':valid_CE_loss,
							'KL_loss':valid_KL_loss,
							'accuracy' : valid_acc}
			torch.save(checkpoint, SAVE_FILE)
			print('[INFO] -> The checkpoint file has been updated.')

def inference(device,model_data):

	generator = Encoder_Decoder(vocab_size=len(model_data.word2index)).to(device)
	generator.load_state_dict(torch.load(SAVE_FILE)["state_dict"])
	loss_func = ModifiedLoss().to(device)

	
	valid_dset = Driver_Data(
		data = model_data.test_prefixes,
		targets = model_data.test_sufixes,
		word2index = model_data.word2index)

	valid_loader = DataLoader(valid_dset, batch_size = 1, shuffle = False, num_workers = 1,collate_fn=pack_collate_fn) #Load Validation data
	generator.eval()
	avg_acc = 0
	counter = 0
	with torch.no_grad():
		for batch in valid_loader:
			counter += 1
			sequences,sequence_lengths,targets, target_lengths = batch
			print(sequences.shape)
			print(targets.shape)
			exit(0)
			sequences = sequences.to(device)
			targets = targets.to(device)

			embedded_inputs = generator.embedding(sequences) #[batch,max_len,emb_dim]
			_, hidden  = generator.encoder(embedded_inputs)
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(0)

			predictions = []
			outputs = []
			curr_state = hidden #[1,batch_size,hidden_size]
			next_word_embedding = generator.embedding(targets[:,0].unsqueeze(1)) #[batch,1,emb_dim]

			while True:
				z, mu, logsigma = generator.reparameterize(curr_state.squeeze(0)) 
				decoder_input = torch.cat([next_word_embedding, z.unsqueeze(1)], -1)
				output, curr_state = generator.decoder(decoder_input, curr_state) #output = [batch_size,1,dec_hid_dim]  
				predicted_output = generator.dense(output)
				softmax_output = F.log_softmax(predicted_output.squeeze(1),dim=-1)
				dec_inp_var = torch.max(softmax_output,dim=-1,keepdim=True)[1] #[batch_size,1]
				predictions.append(dec_inp_var.item())
				next_word_embedding = torch.mm(softmax_output,generator.embedding.weight).unsqueeze(1)
				if dec_inp_var.item() == EOS_ID or len(predictions) > 60 :
					break

			# outputs = torch.cat(outputs,dim =1)
			# accuracy = loss_func.get_accuracy()
			
			
			print("-------------------------------------------")
			print(counter,"/",len(valid_dset))
			print("Input           : "," ".join([model_data.index2word[i] for i in sequences.squeeze().cpu().numpy().tolist()]))
			print("Predicted       : "," ".join([model_data.index2word[i] for i in predictions]))
			print("Target          : "," ".join([model_data.index2word[i] for i in targets.squeeze().cpu().numpy().tolist()][1:]))
			# print("LD_distance     : ",LD_distance)
			# print("LD_similarity   : ",LD_similarity)
			print("-------------------------------------------")
			print()
			# avg_LD_dist += LD_distance
			# avg_LD_sim += LD_similarity
	# print("avg_LD_dist : ",avg_LD_dist/len(valid_dset))
	# print("avg_LD_sim  : ",avg_LD_sim/len(valid_dset))

def pretrain_generator(device,model_data):
	generator = Encoder_Decoder(vocab_size=len(model_data.word2index)).to(device)
	
	print("\nGenerator Parameters :")
	print(generator)
	print(f'The model has {sum(p.numel() for p in generator.parameters() if p.requires_grad):,} trainable parameters\n')

	loss_func = ModifiedLoss().to(device)
	optimizer = optim.Adam(generator.parameters())

	train_dset = Driver_Data(
		data    = model_data.train_prefixes,
		targets = model_data.train_sufixes,
		word2index = model_data.word2index)

	train_loader = DataLoader(train_dset, batch_size = BATCH_SIZE,shuffle = True, num_workers = 10,collate_fn=pack_collate_fn) #Load training data

	valid_dset = Driver_Data(
		data = model_data.test_prefixes,
		targets = model_data.test_sufixes,
		word2index = model_data.word2index)

	valid_loader = DataLoader(valid_dset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn) #Load Validation data
	train_generator(generator, train_loader, valid_loader, optimizer, device, loss_func)

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)
	print("[INFO] -> Loading Preprocessed Data ...")
	model_data = torch.load("../data/Helpdesk/helpdesk.pt")
	print("[INFO] -> Done!")

	#-----------FOR TRAINING------------------------------------
	pretrain_generator(device,model_data)
	#-----------FOR INFERENCE------------------------------------
	inference(device,model_data)
