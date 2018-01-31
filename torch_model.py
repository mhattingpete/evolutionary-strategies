# model imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import numpy as np

import gym

# model class
class esModel(nn.Module):
	def __init__(self,envName):
		super(esModel,self).__init__()
		# enviroment
		self.env = gym.make(envName)
		self.inital_exploration = 0.5
		self.exploration = self.inital_exploration
		self.exploration_dec_steps = 10000000
		# model input size
		self.input_size = self.env.observation_space.shape[0]
		# model output size
		self.output_size = self.env.action_space.n
		# container for hidden layers
		self.hidden = []
		# hidden layer sizes
		self.h_size = [16]
		# define the input layer
		self.fc_1 = nn.Linear(self.input_size,self.h_size[0])
		# define the hidden layers
		for i in range(len(self.h_size)-1):
			self.hidden.append(nn.Linear(self.h_size[i],self.h_size[i+1]))
		# define output layer
		self.fc_out = nn.Linear(self.h_size[-1],self.output_size)
		# define softmax output
		self.softmax = nn.Softmax(dim=1)
		# initialize the weights

	# forward propagate through the network
	def forward(self,inp):
		# redefine input
		x = inp
		# input -> hidden
		x = self.fc_1(x)
		# apply the activation function to the input layer
		x = F.relu(x)

		# hidden -> output
		for l in self.hidden:
			# apply activation function
			x = F.relu(l(x))

		# output layer
		x = self.fc_out(x)
		# apply softmax to the output layer
		x = self.softmax(x)
		return x

	# predict with the model
	def predict(self,inp):
		# normalize the input
		x = inp/(np.linalg.norm(inp)+1e-12)
		if len(x.shape) < 2:
			x = np.expand_dims(x,0)
		x = x.astype(np.float32)
		x = Variable(torch.from_numpy(x),requires_grad=False)
		pred = self(x).data.numpy()
		return pred[0].argmax()

	def extract_grad(self):
		tot_size = self.size
		pvec = np.zeros(tot_size,dtype=np.float32)
		count = 0
		for param in self.parameters():
			sz = param.grad.data.numpy().flatten().shape[0]
			pvec[count:count+sz] = param.grad.data.numpy().flatten()
			count += sz
		return pvec.copy()

	def get_weights_flat(self):
		tot_size = self.size
		pvec = np.zeros(tot_size,dtype=np.float32)
		count = 0
		for param in self.parameters():
			sz = param.data.numpy().flatten().shape[0]
			pvec[count:count+sz] = param.data.numpy().flatten()
			count += sz
		return pvec.copy()

	def set_weights_flat(self,pvec):
		pvec = pvec.astype(np.float32)
		tot_size = self.size
		count = 0
		for param in self.parameters():
			sz = param.data.numpy().flatten().shape[0]
			raw  = pvec[count:count+sz]
			reshaped = raw.reshape(param.data.numpy().shape)
			param.data = torch.from_numpy(reshaped)
			count += sz
		return pvec

	@property
	def size(self):
		count = 0
		for param in self.parameters():
			count += param.data.numpy().size
		return count

	@property
	def num_layers(self):
		count = 0
		for param in self.parameters():
			count += 1
		return count

	def get_action(self,state,training=True):
		if training == True:
			self.exploration = max(0,self.exploration - self.inital_exploration/self.exploration_dec_steps)
			if np.random.uniform() < self.exploration:
				return self.env.action_space.sample()
			else:
				return self.predict(state)
		else:
			return self.predict(state)

	def rollout(self,max_steps,training=True,render=False):
		total_reward = 0.0
		state = self.env.reset()
		for _ in range(max_steps):
			if render == True:
				self.env.render()
			action = self.get_action(state,training=training)
			state,reward,done,_ = self.env.step(action)
			total_reward += reward
			if done: break
		return total_reward