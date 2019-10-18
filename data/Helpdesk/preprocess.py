import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
# from constants import *
from tqdm import tqdm
from datetime import datetime
from collections import Counter


DATA_FILE = "helpdesk.csv"
SAVE_FILE = "helpdesk.pt"
random_seed = 42

class Data_Model:
	def __init__(self, name):

		self.name = name
		self.train_prefixes = []
		self.train_sufixes = []
		self.test_prefixes = []
		self.test_sufixes = []
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
			caseID = row[0].strip()
			activityID = row[1].strip()
			t = time.strptime(row[2].strip(), "%Y-%m-%d %H:%M:%S")
			timestamp = datetime.fromtimestamp(time.mktime(t))
	  
			if caseID in trace_dict.keys() :
				trace_dict[caseID].append([activityID,timestamp])
			else:
				trace_dict[caseID] = [[activityID,timestamp]]


		#All traces are already sorted based on timestamps
		all_traces = self.fix_timestamp(trace_dict)

		# plot_data(all_traces)

		
		self.word2index, self.index2word = self.get_vocab(all_traces)
		train_traces, test_traces = self.split_data(all_traces)

		self.train_prefixes = []
		self.train_sufixes = []
		for trace in train_traces:
			for ind,act in enumerate(trace[1:]):
				pref = trace[:ind+1]
				sufx = trace[ind+1:]
				self.train_prefixes.append(pref)
				self.train_sufixes.append(sufx)

		self.test_prefixes = []
		self.test_sufixes = []
		for trace in test_traces:
			for ind,act in enumerate(trace[1:]):
				pref = trace[:ind+1]
				sufx = trace[ind+1:]
				self.test_prefixes.append(pref)
				self.test_sufixes.append(sufx)

	
		print("[INFO]-> Done.")
		
	#Splitting data into train/test splits
	def split_data(self,all_traces):

		validation_split = 0.3
		
		dataset_size = len(all_traces)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		np.random.seed(random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		
		train_traces = []
		valid_traces = []

		for ind in tqdm(train_indices,desc = " -> Preparing train data") :
			train_traces.append(all_traces[ind])
	
		for ind in tqdm(val_indices,desc = " -> Preparing valid data") :
			valid_traces.append(all_traces[ind])

		return train_traces, valid_traces

	#Creating word2index and index2word
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
	print("Training prefixes :")
	print("Number of datapoints : ",len(model_data.train_prefixes))
	print("Example : ",model_data.train_prefixes[1])
	print()
	print("Training sufixes :")
	print("Number of datapoints : ",len(model_data.train_sufixes))
	print("Example : ",model_data.train_sufixes[1])
	print()
	print("Test prefixes :")
	print("Number of datapoints : ",len(model_data.test_prefixes))
	print("Example : ",model_data.test_prefixes[1])
	print()
	print("Test sufixes :")
	print("Number of datapoints : ",len(model_data.test_sufixes))
	print("Example : ",model_data.test_sufixes[1])
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
	plt.title('Helpdesk')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
	plt.savefig("helpdesk_data.png")

def main():
	
	model_data = Data_Model("helpdesk")
	model_data.read_data()
	print("[INFO] -> Saving file !")
	torch.save(model_data,SAVE_FILE)
	print_stats(model_data)


if __name__ == '__main__':
	main()