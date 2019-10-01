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
from constants import *
from data_process import *
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
# from similarity.damerau import Damerau

# damerau = Damerau()

SAVE_FILE = "../data/trained_models/discriminator_CNN.chkpt"

def train_epoch(generator, discriminator, train_loader, optimizer, device,binary_loss_func):
	generator.eval()
	discriminator.train()
	epoch_loss = 0
	BCE_Loss = 0
	accuracy = 0
	epoch_ppl = 0 
	prob_fake_epoch = 0
	prob_real_epoch = 0

	for batch in tqdm(train_loader, mininterval=2,desc='  - (Training)   ', leave=False):

		sequences,sequence_lengths,targets, target_lengths = batch
		sequences = sequences.to(device)
		sequence_lengths = sequence_lengths.to(device)
		targets = targets.to(device)
		target_lengths = target_lengths.to(device)
		

		mask = torch.gt(targets[:,1:], 0).float()

		outputs,kld,ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
		# outputs,ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
		# outputs,fake_ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0.5)
		fake_responses = ael_responses * (mask.unsqueeze(-1).expand_as(ael_responses))
		fake_responses = fake_responses.detach()
		

		# outputs,real_ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=1)
		# real_responses = real_ael_responses * (mask.unsqueeze(-1).expand_as(real_ael_responses))
		# real_responses = real_responses.detach()

		real_responses = generator.embedding(targets[:,1:]).detach() #[batch_size,max_len,emb_dim]
		real_responses = real_responses * (mask.unsqueeze(-1).expand_as(real_responses))

		embedded_query = generator.embedding(sequences).detach() #[batch_size,max_len,emb_dim]	

		optimizer.zero_grad()


		# prob_real,_ = discriminator(embedded_query, real_responses) #[batch_size, 1]
		# prob_fake,_ = discriminator(embedded_query, fake_responses) #[batch_size, 1]
		# prob_fake_epoch += float(torch.mean(prob_fake).item())
		# prob_real_epoch += float(torch.mean(prob_real).item())
		# predictions = torch.cat((prob_fake.squeeze(1),prob_real.squeeze(1))).to(device)
		# real_values = torch.cat((torch.zeros(prob_fake.shape[0]),torch.ones(prob_real.shape[0]))).to(device)
		# B_loss = binary_loss_func(predictions,real_values)
		# B_loss.backward()
		# optimizer.step()
		# BCE_Loss += float(B_loss.item())

		D_real,_ = discriminator(embedded_query, real_responses) #[batch_size, 1]
		D_fake,_ = discriminator(embedded_query, fake_responses) #[batch_size, 1]

		prob_fake_epoch += float(torch.mean(D_fake).item())
		prob_real_epoch += float(torch.mean(D_real).item())
		D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
		epoch_loss += float(D_loss.item())
		predictions = torch.cat((torch.sigmoid(D_fake).squeeze(1),torch.sigmoid(D_real).squeeze(1))).to(device)
		real_values = torch.cat((torch.zeros(D_fake.shape[0]),torch.ones(D_real.shape[0]))).to(device)
		D_loss.backward()
		epoch_loss += float(D_loss.item())
		for p in discriminator.parameters():
			p.data.clamp_(-0.0001, 0.0001)
		
		acc = accuracy_score(real_values.detach().cpu().numpy(),torch.round(predictions).detach().cpu().numpy())
		accuracy += acc
		
		

	epoch_loss /= len(train_loader)
	epoch_ppl = math.exp(epoch_loss)
	prob_fake_epoch /= len(train_loader)
	prob_real_epoch /= len(train_loader)
	BCE_Loss /= len(train_loader)
	accuracy /= len(train_loader)

	return epoch_loss, epoch_ppl, prob_fake_epoch, prob_real_epoch,BCE_Loss,accuracy

def eval_epoch(generator,discriminator,valid_loader, device,binary_loss_func):
	generator.eval()
	discriminator.eval()
	epoch_loss = 0
	epoch_ppl = 0 
	accuracy = 0
	BCE_Loss = 0
	prob_fake_epoch = 0
	prob_real_epoch = 0

	with torch.no_grad():
		for batch in tqdm(valid_loader, mininterval=2,desc='  - (Validating)   ', leave=False):

			sequences,sequence_lengths,targets, target_lengths = batch
			sequences = sequences.to(device)
			sequence_lengths = sequence_lengths.to(device)
			targets = targets.to(device)
			target_lengths = target_lengths.to(device)

			mask = torch.gt(targets[:,1:], 0).float()
			outputs,kld,ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
			# outputs,fake_ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
			fake_responses = ael_responses * (mask.unsqueeze(-1).expand_as(ael_responses))
			# temp1 = F.softmax(outputs,dim =2)
			# temp2 = torch.max(temp1,dim=2,keepdim=True)[1].squeeze(-1)
			# temp2 = mask * temp2

			# fake_responses = generator.embedding(temp2).detach()

			# outputs,real_ael_responses = generator(sequences,sequence_lengths,targets,target_lengths,tf_ratio=0)
			# real_responses = real_ael_responses * (mask.unsqueeze(-1).expand_as(real_ael_responses))

			real_responses = generator.embedding(targets[:,1:]) #[batch_size,max_len,emb_dim]

			real_responses = real_responses * (mask.unsqueeze(-1).expand_as(real_responses))
			
			embedded_query = generator.embedding(sequences) #[batch_size,max_len,emb_dim]

			# prob_real,_ = discriminator(embedded_query, real_responses) #[batch_size, 1]
			# prob_fake,_ = discriminator(embedded_query, fake_responses) #[batch_size, 1]
			# prob_fake_epoch += float(torch.mean(prob_fake).item())
			# prob_real_epoch += float(torch.mean(prob_real).item())
			# B_loss = binary_loss_func(predictions,real_values)
			# BCE_Loss += float(B_loss.item())
			# epoch_loss += float(B_loss.item())
			# predictions = torch.cat((prob_fake.squeeze(1),prob_real.squeeze(1))).to(device)

			D_real,_ = discriminator(embedded_query, real_responses) #[batch_size, 1]
			D_fake,_ = discriminator(embedded_query, fake_responses) #[batch_size, 1]
			prob_fake_epoch += float(torch.mean(D_fake).item())
			prob_real_epoch += float(torch.mean(D_real).item())
			D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
			epoch_loss += float(D_loss.item())
			predictions = torch.cat((torch.sigmoid(D_fake).squeeze(1),torch.sigmoid(D_real).squeeze(1))).to(device)

			
			real_values = torch.cat((torch.zeros(D_fake.shape[0]),torch.ones(D_real.shape[0]))).to(device)
			acc = accuracy_score(real_values.cpu().numpy(),torch.round(predictions).cpu().numpy())
			accuracy += acc

	epoch_loss /= len(valid_loader)
	epoch_ppl = math.exp(epoch_loss)
	prob_fake_epoch /= len(valid_loader)
	prob_real_epoch /= len(valid_loader)
	BCE_Loss /=len(valid_loader)
	accuracy /= len(valid_loader)


	return epoch_loss, epoch_ppl, prob_fake_epoch, prob_real_epoch,BCE_Loss,accuracy

def train_discriminator(generator,discriminator,train_loader, valid_loader, optimizer, device, binary_loss):

	best_valid_acc = 0
	for epoch_i in range(EPOCH):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_loss,train_ppl, train_pf,train_pr,train_BCE,train_acc= train_epoch(generator,discriminator, train_loader, optimizer, device,binary_loss)
		print('  - (Training)       Loss: {loss: 8.5f} <-> PPL: {ppl: 8.5f} <-> Time Taken : {elapse:3.3f} min <-> prob_fake : {pf:3.5f} <-> prob_real : {pr:3.5f} <-> BCE : {BCE:3.5f} <-> Accuracy : {ACC:3.5f}'.format(
		loss=train_loss,ppl=train_ppl,pf=train_pf,pr=train_pr,BCE=train_BCE,ACC=train_acc,elapse=(time.time()-start)/60))

		# start = time.time()
		# valid_loss,valid_ppl,valid_pf,valid_pr,valid_BCE,valid_acc = eval_epoch(generator, discriminator, valid_loader, device,binary_loss)
		# print('  - (Validating)     Loss: {loss: 8.5f} <-> PPL: {ppl: 8.5f} <-> Time Taken : {elapse:3.3f} min <-> prob_fake : {pf:3.5f} <-> prob_real : {pr:3.5f} <-> BCE : {BCE:3.5f} <-> Accuracy : {ACC:3.5f}'.format(
		# loss=valid_loss,ppl=valid_ppl,pf=valid_pf,pr=valid_pr,BCE=valid_BCE,ACC=valid_acc,elapse=(time.time()-start)/60))
		
		# if valid_acc >= best_valid_acc:
		# 	best_valid_acc = valid_acc
		# 	disc_state_dict = discriminator.state_dict()
		# 	checkpoint = {'state_dict': disc_state_dict,
		# 					'epoch': epoch_i,
		# 					'ppl':valid_ppl,
		# 					'disc_loss':valid_loss,
		# 					'BCE_loss' :valid_BCE,
		# 					'prob_real':valid_pr,
		# 					'prob_fake':valid_pf,
		# 					'accuracy':valid_acc}
		# 	torch.save(checkpoint, SAVE_FILE)
		# 	print('[INFO] -> The checkpoint file has been updated.')

#Not supported
def inference(device,model_data):

	generator = Encoder_Decoder(vocab_size=len(model_data.word2index)).to(device)
	generator.load_state_dict(torch.load("../data/trained_models/encoder_decoder_LSTM.chkpt")["state_dict"])
	discriminator = Discriminator_CNN(128,eval("[1,2]")).to(device)
	discriminator.load_state_dict(torch.load("../data/trained_models/discriminator_LSTM.chkpt")["state_dict"])
	AEL = ApproximateEmbeddingLayer(generator.embedding,device).to(device)

	
	valid_dset = Driver_Data(
		data = model_data.test_data,
		targets = model_data.test_targets,
		word2index = model_data.word2index)

	valid_loader = DataLoader(valid_dset, batch_size = 500, shuffle = False, num_workers = 10,collate_fn=pack_collate_fn) #Load Validation data
	
	binary_loss = nn.BCELoss().to(device)
	valid_loss,valid_ppl,valid_pf,valid_pr,valid_BCE,valid_acc=eval_epoch(generator,discriminator, AEL,valid_loader, device,binary_loss)

	start = time.time()
	print('  - (Validating)     Loss: {loss: 8.5f} <-> PPL: {ppl: 8.5f} <-> Time Taken : {elapse:3.3f} min <-> prob_fake : {pf:3.5f} <-> prob_real : {pr:3.5f} <-> BCE : {BCE:3.5f} <-> Accuracy : {ACC:3.5f}'.format(
		loss=valid_loss,ppl=valid_ppl,pf=valid_pf,pr=valid_pr,BCE=valid_BCE,ACC=valid_acc,elapse=(time.time()-start)/60))


def pretrain_discriminator(device,model_data):
	generator = VarGenerator(vocab_size=len(model_data.word2index)).to(device)
	generator.load_state_dict(torch.load("../data/trained_models/VarGenerator.chkpt")["state_dict"])
	# generator = Encoder_Decoder(vocab_size=len(model_data.word2index)).to(device)
	# generator.load_state_dict(torch.load("../data/trained_models/Encoder_Decoder.chkpt")["state_dict"])
	discriminator = Discriminator_CNN(64,[1,2]).to(device)
	# discriminator = Discriminator_LSTM().to(device)

	
	print("\nDiscriminator Parameters :")
	print(discriminator)
	print(f'The model has {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,} trainable parameters\n')

	binary_loss = nn.BCELoss().to(device)
	optimizer = optim.RMSprop(discriminator.parameters(),lr = 0.05)

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
	train_discriminator(generator,discriminator,train_loader, valid_loader, optimizer, device,binary_loss)

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)
	print("[INFO] -> Loading Preprocessed Data ...")
	model_data = torch.load("../data/model_data_bpi17.pt")
	print("[INFO] -> Done!")

	#-----------FOR TRAINING------------------------------------
	pretrain_discriminator(device,model_data)

	#-----------FOR INFERENCE------------------------------------
	# inference(device,model_data)