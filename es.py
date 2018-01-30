import numpy as np
import os
import pickle
import gym
import ray

from torch_model import esModel

@ray.remote
class Worker:
	def __init__(self,config):
		self.config = config
		self.model = esModel(config['envName'])

	def get_reward(self,weights):
		self.model.set_weights_flat(weights)
		total_reward = self.model.rollout(max_steps=self.config['max_steps'],training=True)
		return total_reward

	def mutate(self,p,method='Plain'):
		if method == 'Plain':
			return self.config['sigma'] * p
		else:
			raise NotImplementedError(method)

	def run(self,weights):
		self.model.set_weights_flat(weights)
		rewards = np.zeros(self.config['pop_size'])
		seed = np.random.randint(low=0,high=1e9)
		rng = np.random.RandomState(seed)
		population = rng.randn(self.config['pop_size'],self.model.size)
		for i in range(self.config['pop_size']):
			delta = self.mutate(population[i],method='Plain')
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

	def get_reward(self,weights):
		self.model.set_weights_flat(weights)
		total_reward = self.model.rollout(max_steps=self.config['max_steps'],training=False)
		return total_reward

	def train(self,num_iters,print_step=10):
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
				self.weights = self.weights + self.config['learning_rate']/(self.config['pop_size']*self.config['sigma']) * grad
			# print the current reward
			if (iteration+1) % print_step == 0:
				curr_reward = self.get_reward(self.weights)
				print('iteration({}) -> reward: {}'.format(iteration+1,curr_reward))
				# check for non-improve in performance
				if curr_reward >= self.config['early_stop_reward'] and (prev_reward-10 <= curr_reward <= prev_reward+10):
					done = True
				prev_reward = curr_reward