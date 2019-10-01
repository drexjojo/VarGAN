import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Data_Model:
	"""
		Class to load the data.
		Load using torch.load("file path")
	"""
	def __init__(self, name):

		self.name = name
		self.train_data = []
		self.train_targets = []
		self.test_data = []
		self.test_targets = []
		self.word2index = {}
		self.index2word = {}
	
class Driver_Data(Dataset):
	"""
	Class to pass data to the model for training & validation.
	Inherits Pytorch Dataset
	"""
	def __init__(self,data,targets,word2index):
		self.data = data
		self.targets = targets
		self.word2index = word2index
		if len(self.targets) != len(self.data):
			print("[INFO] -> ERROR in Driver Data !")
			exit(0)

	def __getitem__(self, index):
		input_seq = self.get_sequence(self.data[index])
		target_seq = self.get_target_sequence(self.targets[index])
		input_seq = np.array(input_seq)
		input_seq = torch.LongTensor(input_seq).view(-1)
		target_seq = np.array(target_seq)
		target_seq = torch.LongTensor(target_seq).view(-1)

		return input_seq, target_seq

	def get_sequence(self, sequence):
		if len(sequence) == 0:
			print("[INFO] -> ERROR empty string found !")
			exit(0)
		indices = []
		for word in sequence:
			if word in self.word2index.keys():
				indices.append(self.word2index[word])
			else:
				indices.append(self.word2index["<UNK>"])
		return indices

	def get_target_sequence(self,sequence):
		indices = [self.word2index["<GO>"]]
		for word in sequence:
			if word in self.word2index.keys():
				indices.append(self.word2index[word])
			else:
				print("UNK found")
				exit(0)
				indices.append(self.word2index["<UNK>"])
		indices.append(self.word2index["<EOS>"])
		return indices

	def __len__(self):
		return len(self.data)

def pack_collate_fn(data):
	def merge(sequences):
		lengths = [len(seq) for seq in sequences]
		padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
		for i, seq in enumerate(sequences):
			end = lengths[i]
			padded_seqs[i, :end] = seq[:end]
		return padded_seqs, lengths

	data.sort(key=lambda x: len(x[0]), reverse=True)
	input_seq,target_seq = zip(*data)
	padded_input, input_lengths = merge(input_seq)
	padded_output, output_lengths = merge(target_seq)

	return padded_input, torch.LongTensor(input_lengths), padded_output, torch.LongTensor(output_lengths)