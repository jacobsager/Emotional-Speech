from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch

class csv_loader_i:
	'''
	Load a CSV that looks like:
		|filename|label|position|data ->|
	Can be any number of classes
	'''
	#def __init__(self, csv_path, classes=['neu', 'ang', 'hap', 'sad', 'fea']):
	def __init__(self, csv_path,  classes=['neutral', 'sad', 'angry', 'fear', 'happy']):
		self.raw_data = pd.read_csv(csv_path, header=None)
		#self.filename_arr = np.asarray(self.raw_data.iloc[:, 0])
		self.label_arr = np.asarray(self.raw_data.iloc[:, 0])
		self.data_arr = np.asarray(self.raw_data.iloc[:, 2:])
		self.pos = np.asarray(self.raw_data.iloc[:, 1])
		self.data_len = len(self.raw_data.index)
		self.classes = classes
		
	def __getitem__(self, index):
		outclasses = [0] * len(self.classes)
		single_feature_vector = self.data_arr[index, :]
		#single_label = self.classes.index(self.label_arr[index])
		single_label = self.label_arr[index]
		outclasses[single_label] = 1
		outclasses = np.asarray(outclasses)

		single_feature_vector = torch.from_numpy(single_feature_vector)
		outclasses = torch.from_numpy(outclasses)
		pos = self.pos[index]
		#filename = self.filename_arr[index]

		return (single_feature_vector, outclasses, pos)

	def __len__(self):
		return self.data_len
		
class csv_loader_ii:
	'''
	Load a CSV that looks like:
		|label|position|data ->|
	Can be any number of classes
	'''
	def __init__(self, csv_path,  classes=['neutral', 'sad', 'angry', 'fear', 'happy']):
		self.raw_data = pd.read_csv(csv_path, header=None)
		self.label_arr = np.asarray(self.raw_data.iloc[:, 0])
		self.data_arr = np.asarray(self.raw_data.iloc[:, 2:])
		self.pos = np.asarray(self.raw_data.iloc[:, 1])
		self.data_len = len(self.raw_data.index)
		self.classes = classes
		
	def __getitem__(self, index):
		outclasses = [0] * len(self.classes)
		single_feature_vector = self.data_arr[index, :]
		single_label = self.classes.index(self.label_arr[index])
		outclasses[single_label] = 1
		outclasses = np.asarray(outclasses)

		single_feature_vector = torch.from_numpy(single_feature_vector)
		outclasses = torch.from_numpy(outclasses)
		pos = self.pos[index]

		return (single_feature_vector, outclasses, pos, pos)

	def __len__(self):
		return self.data_len