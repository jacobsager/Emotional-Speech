import torch.nn as nn

class three_layer_net(nn.Module):
	def __init__(self, input_size, hidden1_size, num_classes, batch_size):
		super(three_layer_net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden1_size)
		self.fc2 = nn.Linear(hidden1_size, hidden1_size)
		self.fc3 = nn.Linear(hidden1_size, num_classes)
		self.relu = nn.Sigmoid()
		self.sm = nn.LogSoftmax(dim=1)
		self.dropout = nn.Dropout(.2)
		self.leaky_relu = nn.LeakyReLU()
		self.batch_size = batch_size
		self.input_size = input_size
		self.batch_norm1 = nn.BatchNorm1d(input_size)
		self.batch_norm2 = nn.BatchNorm1d(hidden1_size)
		self.batch_norm3 = nn.BatchNorm1d(hidden1_size)
  
	def forward(self, x):
		x = x.view(self.batch_size,self.input_size)
		#x = self.batch_norm1(x)
		
		out = self.fc1(x)
		#out = self.dropout(out)
		out = self.batch_norm2(out)
		out=self.leaky_relu(out)

		out = self.fc2(out)
		#out = self.dropout(out)
		out = self.batch_norm3(out)
		out=self.leaky_relu(out)
		
		
		out = self.fc3(out)
		out=self.leaky_relu(out)
		
		return out
		
class four_layer_net(nn.Module):
	def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
		super(four_layer_net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden1_size)
		self.fc2 = nn.Linear(hidden1_size, hidden2_size)
		self.fc3 = nn.Linear(hidden2_size, num_classes)
		self.relu = nn.Sigmoid()
		self.sm = nn.LogSoftmax(dim=1)
		self.dropout = nn.Dropout(.2)
  
	def forward(self, x):
		out = self.relu(self.fc1(x))	
		out = self.relu(self.fc2(out))
		out = self.sm(self.fc3(out))
		return out
		
class five_layer_net(nn.Module):
	def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, num_classes):
		super(five_layer_net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden1_size)
		self.fc2 = nn.Linear(hidden1_size, hidden2_size)
		self.fc3 = nn.Linear(hidden2_size, hidden3_size)
		self.fc4 = nn.Linear(hidden3_size, num_classes)
		self.relu = nn.Sigmoid()
		self.sm = nn.LogSoftmax(dim=1)
		self.dropout = nn.Dropout()
  
	def forward(self, x):
		out = self.relu(self.dropout(self.fc1(x)))	
		out = self.relu(self.dropout(self.fc2(out)))
		out = self.relu(self.dropout(self.fc3(out)))
		out = self.sm(self.dropout(self.fc4(out)))
		return out
		
class ELM(nn.Module):
	def __init__(self):
		super(ELM, self).__init__()
		#input length 4 for max, min, mean, percentage threshold stats of all output frames per utterance for each emotion
		self.fc1 = nn.Linear(4*5, 120)
		self.fc2 = nn.Linear(120, 5)
		self.relu = nn.Sigmoid()
		self.sm = nn.LogSoftmax(dim=1)
		
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.sm(out)
		
		return out