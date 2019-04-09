import neural_nets
import loaders
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()

class three_layer_net:
	def __init__(self, train_file_path, test_file_path, input_layer=144, hidden_layer=90, output_layer=5, saved_model=None, batch_size=1):
		self.results = Results()
		self.net = neural_nets.three_layer_net(input_layer, hidden_layer, output_layer, batch_size)
		if use_cuda:
			self.net = self.net.cuda(0)
			self.net = torch.nn.DataParallel(self.net, device_ids=[0])
		self.results.num_emotions = output_layer
		if saved_model: self.net.load_state_dict(torch.load(saved_model))
		self.results.architecture = self.net
		self.train_file_path = train_file_path
		self.test_file_path = test_file_path
		self.batch_size = batch_size
		
	def train(self, shuffle=True, epochs=50, criterion=nn.CrossEntropyLoss(), lr=.00007, loader='i', save_filename=None, weights=[1, 1, 1, 1, 1]):
		self.results.loss = __train__(self.net, self.train_file_path, self.test_file_path, shuffle, epochs, criterion, lr, loader, save_filename, weights, self.batch_size)
		
	def test(self, loader='i', shuffle=False):
		self.results = __test__(self.test_file_path, self.net, shuffle, loader, self.results, self.batch_size)
	
	def get_results(self): return self.results
	
class four_layer_net:
	def __init__(self, input_layer=144, h1=144, h2=90, output_layer=6, saved_model=None):
		self.results = Results()
		self.net = neural_nets.four_layer_net(input_layer, h1, h2, output_layer)
		self.results.num_emotions = output_layer
		if saved_model: self.net.load_state_dict(torch.load(saved_model))
		self.results.architecture = self.net
		
	def train(self, train_file_path, shuffle=True, epochs=50, criterion=nn.CrossEntropyLoss(), lr=.00007, loader='i', save_filename=None):
		self.results.loss = __train__(self.net, train_file_path, shuffle, epochs, criterion, lr, loader, save_filename)
		
	def test(self, test_file_path, loader='i', shuffle=False):
		self.results = __test__(test_file_path, self.net, shuffle, loader, self.results)
	
	def get_results(self): return self.results
	
class five_layer_net:
	def __init__(self, input_layer=144, h1=120, h2=90, h3=40, output_layer=6, saved_model=None):
		self.results = Results()
		self.net = neural_nets.five_layer_net(input_layer, h1, h2, h3, output_layer)
		self.results.num_emotions = output_layer
		if saved_model: self.net.load_state_dict(torch.load(saved_model))
		self.results.architecture = self.net
		
	def train(self, train_file_path, shuffle=True, epochs=50, criterion=nn.CrossEntropyLoss(), lr=.00007, loader='i', save_filename=None):
		self.results.loss = __train__(self.net, train_file_path, shuffle, epochs, criterion, lr, loader, save_filename)
		
	def test(self, test_file_path, loader='i', shuffle=False):
		self.results = __test__(test_file_path, self.net, shuffle, loader, self.results)
	
	def get_results(self): return self.results
	
class elm_net:
	def __init__(self, input_layer=144, hidden_layer=90, output_layer=6, saved_model=None):
		self.results = Results()
		self.results.num_emotions = output_layer
		self.net = neural_nets.three_layer_net(input_layer, hidden_layer, output_layer)
		self.elm = neural_nets.ELM()
		self.results.architecture = self.net
	
	def train(self, train_file_path, shuffle=True, epochs=50, criterion=nn.CrossEntropyLoss(), lr=.00007, loader='i', save_filename=None):
		__train__(self.net, train_file_path, shuffle, epochs, criterion, lr, loader, save_filename)
		self.results.loss = __train_ELM__(net, elm_net, train_file_path, shuffle, epochs, criterion, lr, loader, save_filename)
	
	def test(self, test_file_path, loader='i', shuffle=False):
		self.results = __test__(test_file_path, self.net, shuffle, loader, self.results)
	
	def get_results(self): return self.results
	
'''
train/test generic functions below
'''
def __train__(net, train_file_path, test_file_path, shuffle, epochs, criterion, lr, loader, save_filename, weights, batch_size):
	if loader == 'i': trainloader = torch.utils.data.DataLoader(loaders.csv_loader_i(train_file_path), shuffle=shuffle, batch_size=batch_size, drop_last=True)
	if loader == 'ii': trainloader = torch.utils.data.DataLoader(loaders.csv_loader_ii(train_file_path), shuffle=shuffle)
	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=.00001)
	lss = []

	#class_weights = torch.FloatTensor([.078, .187, .313, .143, .278])
	#class_weights = torch.FloatTensor([1, 1, 1, 1, 1])
	class_weights = torch.FloatTensor(weights)
	if use_cuda:
		class_weights = class_weights.cuda()

	criterion=nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
	net.train()
	for epoch in range(epochs):
		results = Results()
		results = __test__(test_file_path, net, False, 'i', results, batch_size)
		test_acc = results.accuracy
		test_acc_c = results.accuracy_per_category
		results = Results()
		results = __test__(train_file_path, net, False, 'i', results, batch_size)
		train_acc = results.accuracy
		print('Epoch ' + str(epoch) + ': Train Accuracy=' + str(round(train_acc, 4)) + ' Test Acc='+ str(round(test_acc,4)) + ' ' + str(test_acc_c))
		net.train()
		running_loss = 0.0

		for i, (data, labels, pos) in enumerate(trainloader):
			if use_cuda:
				data, labels = data.cuda(), labels.cuda()
			data, labels = Variable(data.float()), Variable(labels.long())
			optimizer.zero_grad()

			# Forward pass
			#outputs = net(data)
			outputs = net(data).view(batch_size, 5)
			loss = criterion(outputs, torch.max(labels, 1)[1])
			#loss = weighted_cross_entropy_loss(outputs, labels)

			# Backward and optimize
			loss.backward()
			optimizer.step()
			running_loss += loss.data.item()

			if i % 10000 == 9999: 
				lss.append(running_loss/10000)
				#print('Epoch: ' + str(epoch) + ', Loss: ' + str(round(running_loss/1000,4)))
				running_loss = 0.0
	if save_filename: torch.save(net.state_dict(), save_filename)
	#print('Training Complete')
	return lss
	
def __test__(test_file_path, net, shuffle, loader, results, batch_size):
	net.eval()
	all_out=[]
	out_size = 5
	correct, total = 0, 0
	ind_total, ind_correct, guess = [0] * out_size, [0] * out_size, [0] * out_size
	conf = np.zeros((out_size + 1, out_size + 1))
	predicted_arr = []
	last = -1
	cur_label = None
	if loader == 'i': testloader = torch.utils.data.DataLoader(loaders.csv_loader_i(test_file_path), shuffle=shuffle, batch_size=batch_size, drop_last=True)
	if loader == 'ii': testloader = torch.utils.data.DataLoader(loaders.csv_loader_ii(test_file_path), shuffle=shuffle)
	
	
	for i, (data, l, position) in enumerate(testloader):
		
		data, l = Variable(data.float()), Variable(l.long())
		if use_cuda:
			data, l = data.cuda(), l.cuda()
		op = F.softmax(net(data), dim=1)
			
		for j in range(batch_size):
			outputs, labels, pos = op[j], l[j], position[j]
			#data, labels = Variable(data.float()), Variable(labels.long())
			#data, labels = Variable(current_x.float()), Variable(current_label.long())
			
			val, predicted = torch.max(outputs, 0)
			val, predicted = val.item(), predicted.item()
			actual = torch.max(labels, 0)[1]
		
			if i == 0:
				last = pos.item()
				cur_label = actual
				
			if last == pos.item():
				guess = np.add(guess, outputs.tolist())
				last = pos.item()
			else:
				all_out.append(list(guess))
				#correct?
				predicted = np.argmax(guess)
				conf[predicted + 1, cur_label.item() + 1] = conf[predicted + 1, cur_label.item() + 1] + 1
				
				total += 1
				ind_total[cur_label.item()] += 1
				
				if predicted == cur_label.item():
					correct += 1
					ind_correct[cur_label.item()] += 1

				guess = [0] * out_size
				guess = np.add(guess, outputs.tolist())
				last = pos.item()
				cur_label = actual
			
	ind_percentages = [0.0] * out_size
	for i in range(len(ind_percentages)):
		ind_percentages[i] = round(100*(ind_correct[i]/(ind_total[i]+1)))

	conf = conf.astype(int)
	conf = conf.astype(str)
	results.accuracy_per_category = ind_percentages
	results.accuracy = (100 * correct / total)
	results.conf_matrix = conf
	#print('Testing Complete')
	#print(all_out)
	return results

'''
Output object
'''
class Results:
	def __init__(self):
		self.num_emotions = None
		self.architecture = None
		self.loss = []
		self.accuracy = None
		self.accuracy_per_category = None
		self.conf_matrix = None
	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)
