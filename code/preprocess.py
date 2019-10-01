import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from constants import *
from tqdm import tqdm
from datetime import datetime
from collections import Counter


DATA_FILE = "../data/bpi_17.csv"
SAVE_FILE = "../data/model_data_bpi17.pt"
random_seed = 42

class Data_Model:
	def __init__(self, name):

		self.name = name
		self.train_data = []
		self.train_targets = []
		self.test_data = []
		self.test_targets = []
		self.word2index = {}
		self.index2word = {}

	#Fix trace according to timestamp
	def fix_timestamp(self,cascade_dict):
		all_traces = []
		for key,value in cascade_dict.items():
			cascade_dict[key].sort(key=lambda x:x[1])
			trace = [i[0] for i in cascade_dict[key]]
			all_traces.append(trace)

		return all_traces

	def read_data(self):

		data = []
		with open(DATA_FILE) as f:
			for line in f.readlines()[1:]:
				lis = line.split(",")
				data.append(lis)

		trace_dict = {}
		for row in tqdm(data,desc = " -> Reading data") :
			application = row[0].strip()
			event = row[1].strip()
			t = time.strptime(row[2].strip(), "%Y-%m-%d %H:%M:%S")
			timestamp = datetime.fromtimestamp(time.mktime(t))
	  
			if application in trace_dict.keys() :
				trace_dict[application].append([event,timestamp])
			else:
				trace_dict[application] = [[event,timestamp]]


		all_traces = self.fix_timestamp(trace_dict)

		# plot_data(all_traces)

		all_data = []
		all_targets = []

		for trace in all_traces:
			mid = len(trace)//2
			for i in range(4):
				split_point = mid+i
				all_data.append(trace[:split_point])
				all_targets.append(trace[split_point:])
		
		self.word2index, self.index2word = self.get_vocab(all_traces)
		self.train_data, self.train_targets, self.test_data, self.test_targets = self.split_data(all_data,all_targets)
	
		print("[INFO]-> Done.")
		
	#Splitting data into train/test splits
	def split_data(self,all_data,all_targets):

		validation_split = 0.2
		
		dataset_size = len(all_data)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		np.random.seed(random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		
		train_data = []
		valid_data = []
		train_targets = []
		valid_targets = []

		for ind in tqdm(train_indices,desc = " -> Preparing train data") :
			train_data.append(all_data[ind])
			train_targets.append(all_targets[ind])

		for ind in tqdm(val_indices,desc = " -> Preparing valid data") :
			valid_data.append(all_data[ind])
			valid_targets.append(all_targets[ind])

		return train_data, train_targets, valid_data, valid_targets

	#Creating wod2index and index2word
	def get_vocab(self,traces):
		all_words = []
		for seq in traces :
			for i in seq :
				all_words.append(i)

		vocab =  ['<PAD>', '<GO>', '<EOS>'] + list(set(all_words))
		word2index = {word:i for i,word in enumerate(vocab)}
		index2word = {i:word for word,i in word2index.items()}
		
		return word2index, index2word
		
def print_stats(model_data):
	print("Training data :")
	print("Number of datapoints : ",len(model_data.train_data))
	print("Example : ",model_data.train_data[1])
	print()
	print("Training targets :")
	print("Number of datapoints : ",len(model_data.train_targets))
	print("Example : ",model_data.train_targets[1])
	print()
	print("Test data :")
	print("Number of datapoints : ",len(model_data.test_data))
	print("Example : ",model_data.test_data[1])
	print()
	print("Test targets :")
	print("Number of datapoints : ",len(model_data.test_targets))
	print("Example : ",model_data.test_targets[1])
	print()
	print("Word2Index :")
	print("Number of datapoints : ",len(model_data.word2index))
	print()
	print("Index2Word :")
	print("Number of datapoints : ",len(model_data.index2word))
	print()

def plot_data(all_traces):
	lengths = [len(x) for x in all_traces]

	n, bins, patches = plt.hist(x=lengths, bins='auto', color='#0504aa',alpha=0.9, rwidth=1)
	plt.grid(axis='y', alpha=0.9)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('Flixter')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
	plt.savefig("Flixter_data.png")

def main():
	
	model_data = Data_Model("bpi17")
	model_data.read_data()
	print("[INFO] -> Saving file !")
	torch.save(model_data,SAVE_FILE)
	print_stats(model_data)
	# plot_data()

if __name__ == '__main__':
	main()