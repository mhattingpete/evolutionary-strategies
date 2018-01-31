# for parallel computing
import ray

# the agent to train the model
from es import Master

# number of cpus to be used
num_cpus = 3

# start the redis server
ray.init(num_cpus=num_cpus)

# config to run
config = {'sigma':0.1,'pop_size':50,'l2_coeff':0.00005,'num_workers':num_cpus,'max_steps':200,'learning_rate':0.001,
'envName':'CartPole-v0','early_stop_reward':200}

# run the training procedure
def train():
	# define the agent
	agent = Master(config)
	# run the agent for a total of num_iters iterations
	agent.train(num_iters=1000,print_step=10)
	# run the agent on the enviroment and render
	avg_reward = agent.play(episodes=10)
	# filename to save the weights in
	filename = config['envName']+'_weights_'+str(avg_reward)
	# save the weights
	agent.save(filename)

def trainR():
	avg_reward = 0.0
	agent = None
	while avg_reward != config['early_stop_reward']:
		del agent
		# define the agent
		agent = Master(config)
		# run the agent for a total of num_iters iterations
		agent.train(num_iters=1000,print_step=10)
		# run the agent on the enviroment and render
		avg_reward = agent.play(episodes=10)
	# filename to save the weights in
	filename = config['envName']+'_weights_'+str(avg_reward)
	# save the weights
	agent.save(filename)

if __name__ == '__main__':
	trainR()