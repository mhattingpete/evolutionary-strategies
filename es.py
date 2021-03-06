import numpy as np
import os
import pickle
import gym
import ray
import time
import random

import torch
from torch.autograd import Variable, grad

from scipy.optimize import minimize_scalar, minimize

from torch_model import esModel
import optimizers

@ray.remote
class Worker:
	def __init__(self,config):
		self.config = config
		self.model = esModel(config['envName'])
		env = gym.make(config['envName'])
		self.v_states = [env.reset()]
		for _ in range(1000):
			s,_,d,_ = env.step(env.action_space.sample())
			self.v_states.append(s)
			if d: break
		del env

	def get_reward(self,weights):
		self.model.set_weights_flat(weights)
		total_reward = self.model.rollout(max_steps=self.config['max_steps'],training=True)
		return total_reward

	def mutate(self,p,old_w,method='Plain'):
		if method == 'Plain':
			return self.config['sigma'] * p
		elif method == 'SM-G-SO':
			self.model.set_weights_flat(old_w)
			sz = min(100,len(self.v_states))
			v_states = np.array(random.sample(self.v_states,sz),dtype=np.float32)
			v_states = v_states / (np.linalg.norm(v_states)+1e-12)
			v_states = Variable(torch.from_numpy(v_states),requires_grad=False)
			old_policy = self.model(v_states)

			delta = self.config['sigma'] * p
			np_copy = np.array(old_policy.data.numpy(),dtype=np.float32)
			_old_policy_cached = Variable(torch.from_numpy(np_copy),requires_grad=False)
			loss = ((old_policy-_old_policy_cached)**2).sum(1).mean(0)
			loss_gradient = grad(loss,self.model.parameters(),create_graph=True)
			flat_gradient = torch.cat([grads.view(-1) for grads in loss_gradient])

			direction = (delta/np.sqrt((delta**2).sum())).astype(np.float32)
			direction_t = Variable(torch.from_numpy(direction),requires_grad=False)
			grad_v_prod = (flat_gradient * direction_t).sum()
			second_deriv = grad(grad_v_prod,self.model.parameters())
			sensitivity = torch.cat([g.contiguous().view(-1) for g in second_deriv])
			scaling = torch.sqrt(torch.abs(sensitivity).data)
			scaling = scaling.numpy()
			scaling[scaling==0]=1.0
			scaling[scaling<0.01]=0.01
			delta /= scaling
			return np.clip(delta,-3,3)
		elif method == 'R':
			self.model.set_weights_flat(old_w)
			sz = min(100,len(self.v_states))
			v_states = np.array(random.sample(self.v_states,sz),dtype=np.float32)
			v_states = v_states / (np.linalg.norm(v_states)+1e-12)
			v_states = Variable(torch.from_numpy(v_states),requires_grad=False)
			old_policy = self.model(v_states)

			# a fast method which does often leads to a better performance
			search_rounds = 7
			delta = p
			delta = delta / np.sqrt((delta**2).sum())
			threshold = self.config['sigma']
			old_policy = np.array(old_policy.data.numpy(),dtype=np.float32)
			def search_error(x):
				final_delta = delta * x
				final_delta = np.clip(final_delta,-3,3)
				new_w = old_w + final_delta
				self.model.set_weights_flat(new_w)
				output = self.model(v_states).data.numpy().astype(np.float32)
				change = np.sqrt(((output - old_policy)**2).sum(1)).mean()
				return np.sqrt(change-threshold)**2
			mult = minimize_scalar(search_error,bounds=(0,self.config['sigma'],3),tol=0.01**2,options={'maxiter':search_rounds})
			delta *= mult.x
			return np.clip(delta,-3,3)
		elif method == 'R-ALT':
			self.model.set_weights_flat(old_w)
			sz = min(100,len(self.v_states))
			v_states = np.array(random.sample(self.v_states,sz),dtype=np.float32)
			v_states = v_states / (np.linalg.norm(v_states)+1e-12)
			v_states = Variable(torch.from_numpy(v_states),requires_grad=False)
			old_policy = self.model(v_states)

			# this method is really slow and doens't seem to improve performance
			search_rounds = 7
			delta = p
			delta = delta / np.sqrt((delta**2).sum())
			threshold = self.config['sigma']
			old_policy = np.array(old_policy.data.numpy(),dtype=np.float32)
			new_w = np.zeros(self.model.size)
			def search_error_alt(x):
				count = 0
				final_delta = delta
				for i,p in enumerate(self.model.parameters()):
					sz = p.data.numpy().flatten().shape[0]
					final_delta[count:count+sz] = delta[count:count+sz] * x[i]
					count += sz
				new_w = old_w + final_delta
				self.model.set_weights_flat(new_w)
				output = self.model(v_states).data.numpy().astype(np.float32)
				change = np.sqrt(((output-old_policy)**2).sum(1)).mean()
				return np.sqrt(change-threshold)**2
			bounds = np.ones((self.model.num_layers,2))
			bounds[:,0] = 1e-6*bounds[:,0]
			bounds[:,1] = 3*bounds[:,0]
			x0 = self.config['sigma']*np.ones(self.model.num_layers)
			mult = minimize(search_error_alt,x0=x0,tol=0.01**2,bounds=bounds,options={'maxiter':search_rounds})
			count = 0
			for i,p in enumerate(self.model.parameters()):
				sz = p.data.numpy().flatten().shape[0]
				delta[count:count+sz] *= mult.x[i]
				count += sz
			return np.clip(delta,-3,3)
		else:
			raise NotImplementedError(method)

	def run(self,weights):
		self.model.set_weights_flat(weights)
		rewards = np.zeros(self.config['pop_size'])
		seed = np.random.randint(low=0,high=1e9)
		rng = np.random.RandomState(seed)
		population = rng.randn(self.config['pop_size'],self.model.size)
		for i in range(self.config['pop_size']):
			delta = self.mutate(population[i],weights,method='R')
			new_weights = weights + delta
			rewards[i] = self.get_reward(new_weights)
		# normalize rewards
		rewards = (rewards - np.mean(rewards))/(np.std(rewards)+1e-12)
		del new_weights
		del population
		return [rewards,seed]

class Master:
	def __init__(self,config):
		self.config = config
		self.model = esModel(config['envName'])
		self.optimizer = optimizers.Adam(self.model.get_weights_flat(),self.config['learning_rate'],beta1=0.9,beta2=0.999,epsilon=1e-08)

	def get_reward(self,weights):
		self.model.set_weights_flat(weights)
		total_reward = self.model.rollout(max_steps=self.config['max_steps'],training=False)
		return total_reward

	def play(self,episodes=10):
		self.model.set_weights_flat(self.weights)
		rewards = np.zeros(episodes)
		for i in range(episodes):
			rewards[i] = self.model.rollout(max_steps=self.config['max_steps'],training=False,render=False)
		avg_reward = rewards.mean()
		print('Reward for each episode was:\n',rewards)
		print('Average reward across {} episodes: {}'.format(episodes,avg_reward))
		return avg_reward

	def save(self,filename):
		filename = os.path.join('Checkpoints',filename +'.pkl')
		if not os.path.isdir('Checkpoints'):
			os.mkdir('Checkpoints')
		with open(filename,'wb') as fp:
			pickle.dump(self.weights,fp)

	def load(self,filename):
		filename = os.path.join('Checkpoints',filename +'.pkl')
		with open(filename,'rb') as fp:
			self.model.set_weights_flat(pickle.load(fp))

	def train(self,num_iters,print_step=10):
		start = time.time()
		done = False
		prev_reward = 0
		self.weights = self.model.get_weights_flat()
		self.workers = [Worker.remote(self.config) for _ in range(self.config['num_workers'])]
		for iteration in range(num_iters):
			if done: break
			results = ray.get([worker.run.remote(self.weights) for worker in self.workers])
			# get the results for every worker
			for res in results:
				# collect rewards
				rewards = np.array(res[0])
				seed = res[1]
				rng = np.random.RandomState(seed)
				# reconstruct population for the worker
				population = rng.randn(self.config['pop_size'],self.model.size)
				# compute the gradient
				grad = np.dot(rewards.T,population).T
				# update the weights
				#self.weights = self.weights + self.config['learning_rate']/(self.config['pop_size']*self.config['sigma']) * grad
				grad = 1.0/(self.config['pop_size']*self.config['sigma']) * grad
				_, self.weights = self.optimizer.update(-grad)
			# print the current reward
			if (iteration+1) % print_step == 0:
				curr_reward = self.get_reward(self.weights)
				print('iteration({}) -> reward: {}'.format(iteration+1,curr_reward))
				# check for non-improve in performance
				if curr_reward >= self.config['early_stop_reward'] and (prev_reward-10 <= curr_reward <= prev_reward+10):
					done = True
				prev_reward = curr_reward
		print('Training took {}s'.format(time.time()-start))