import argparse
import os
import sys
import random
import numpy as np
 
from params import Params
from process_mocap import build_KD_tree, tree_query, coordinate_transform

from scipy.spatial.transform import Rotation
 
import pickle
import time
 
import statistics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from operator import add, sub
 
import pickle
 
 
class PPOStorage:
	def __init__(self, num_inputs, num_outputs, max_size=64000):
		self.states = torch.zeros(max_size, num_inputs).to(device)
		self.next_states = torch.zeros(max_size, num_inputs).to(device)
		self.actions = torch.zeros(max_size, num_outputs).to(device)
		self.dones = torch.zeros(max_size, 1, dtype=torch.int8).to(device)
		self.log_probs = torch.zeros(max_size).to(device)
		self.rewards = torch.zeros(max_size).to(device)
		self.q_values = torch.zeros(max_size, 1).to(device)
		self.mean_actions = torch.zeros(max_size, num_outputs).to(device)
		self.counter = 0
		self.sample_counter = 0
		self.max_samples = max_size
	def sample(self, batch_size):
		idx = torch.randint(self.counter, (batch_size,),device=device)
		return self.states[idx, :], self.actions[idx, :], self.next_states[idx, :], self.rewards[idx], self.q_values[idx, :], self.log_probs[idx]
	def clear(self):
		self.counter = 0
	def push(self, states, actions, next_states, rewards, q_values, log_probs, size):
		self.states[self.counter:self.counter+size, :] = states.detach().clone()
		self.actions[self.counter:self.counter+size, :] = actions.detach().clone()
		self.next_states[self.counter:self.counter+size, :] = next_states.detach().clone()
		self.rewards[self.counter:self.counter+size] = rewards.detach().clone()
		self.q_values[self.counter:self.counter+size, :] = q_values.detach().clone()
		self.log_probs[self.counter:self.counter+size] =  log_probs.detach().clone()
		self.counter += size

	def discriminator_sample(self, batch_size):
		if self.sample_counter == 0 or self.sample_counter == self.max_samples:
			self.permute()
		self.sample_counter %= self.max_samples
		self.sample_counter += batch_size
		return self.states[self.sample_counter-batch_size:self.sample_counter, :], self.next_states[self.sample_counter-batch_size:self.sample_counter, :]

	def critic_sample(self, batch_size):
		if self.sample_counter == 0 or self.sample_counter == self.max_samples:
			self.permute()
		self.sample_counter %= self.max_samples
		self.sample_counter += batch_size
		return self.states[self.sample_counter-batch_size:self.sample_counter, :], self.q_values[self.sample_counter-batch_size:self.sample_counter, :]
	
	def actor_sample(self, batch_size):
		if self.sample_counter == 0 or self.sample_counter == self.max_samples:
			self.permute()
		self.sample_counter %= self.max_samples
		self.sample_counter += batch_size
		return self.states[self.sample_counter-batch_size:self.sample_counter, :], self.actions[self.sample_counter-batch_size:self.sample_counter, :], self.next_states[self.sample_counter-batch_size:self.sample_counter, :], self.q_values[self.sample_counter-batch_size:self.sample_counter, :], self.log_probs[self.sample_counter-batch_size:self.sample_counter]

	def permute(self):
		permuted_index = torch.randperm(self.max_samples)
		self.states[:, :] = self.states[permuted_index, :]
		self.next_states[:, :] = self.states[permuted_index, :]
		self.actions[:, :] = self.actions[permuted_index, :]
		self.q_values[:, :] = self.q_values[permuted_index, :]
		self.log_probs[:] = self.log_probs[permuted_index]
 
class RL(object):
	def __init__(self, env, hidden_layer=[64, 64]):
		self.env = env
		#self.env.env.disableViewer = False
		self.num_inputs = env.observation_space.shape[0]  #84 for kin
		self.num_outputs = env.action_space.shape[0]  #9 for kin
		self.hidden_layer = hidden_layer
 
		self.params = Params()
		self.Net = ActorCriticNetMann
		self.model = self.Net(self.num_inputs, self.num_outputs,self.hidden_layer)
		self.discriminator = Discriminator(75 * 2, [128, 128]).to(device)
		self.model.share_memory()
		self.test_mean = []
		self.test_std = []
 
		self.noisy_test_mean = []
		self.noisy_test_std = []
		self.fig = plt.figure()
		#self.fig2 = plt.figure()
		self.lr = self.params.lr
		plt.show(block=False)
 
		self.test_list = []
		self.noisy_test_list = []
 
		self.best_score_queue = mp.Queue()
		self.best_score = mp.Value("f", 0)
		self.max_reward = mp.Value("f", 5)
 
		self.best_validation = 1.0
		self.current_best_validation = 1.0

		self.train_dynamics = True

		if self.train_dynamics:
			self.gpu_model = self.Net(self.num_inputs, self.num_outputs,self.hidden_layer)
			self.model_old = self.Net(self.num_inputs, self.num_outputs, self.hidden_layer).to(device)
		else:
			self.gpu_model = self.Net(84, 9, self.hidden_layer).to(device)  #82 69
			self.model_old = self.Net(84, 9, self.hidden_layer).to(device)
		self.kinematic_policy = ActorCriticNetMann(84, 9, self.hidden_layer).to(device)
		# self.dynamic_policy = ActorCriticNet(self.num_inputs, self.num_outputs,[1024, 1024]).to(device)
		# self.dynamic_policy.load_state_dict(torch.load("stats/dynamics_FF_ELU/iter199999.pt"))
		self.gpu_model.to(device)

 
		self.base_controller = None
		self.base_policy = None
 
		self.total_rewards = []

		self.critic_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)
		self.actor_optimizer = optim.AdamW(self.gpu_model.parameters(), lr=1e-4)

 
	def run_test_with_noise(self, num_test=10):
 
		reward_mean = statistics.mean(self.total_rewards)
		reward_std = statistics.stdev(self.total_rewards)
		self.noisy_test_mean.append(reward_mean)
		self.noisy_test_std.append(reward_std)
		self.noisy_test_list.append((reward_mean, reward_std))
 
		print("reward mean,", reward_mean)
		print("reward std,", reward_std)
 
	def save_reward_stats(self, stats_name):
		with open( stats_name, 'wb') as f:
			np.save(f, np.array(self.noisy_test_mean))
			np.save(f, np.array(self.noisy_test_std))
 
	def plot_statistics(self):
		plt.clf()
		ax = self.fig.add_subplot(121)
		#ax2 = self.fig.add_subplot(122)
		low = []
		high = []
		index = []
		noisy_low = []
		noisy_high = []
		for i in range(len(self.noisy_test_mean)):
			noisy_low.append(self.noisy_test_mean[i]-self.noisy_test_std[i])
			noisy_high.append(self.noisy_test_mean[i]+self.noisy_test_std[i])
			index.append(i)
		plt.xlabel('iterations')
		plt.ylabel('average rewards')
		#ax.plot(self.test_mean, 'b')
		ax.plot(self.noisy_test_mean, 'g')
		#ax.fill_between(index, low, high, color='cyan')
		ax.fill_between(index, noisy_low, noisy_high, color='r')
		#ax.plot(map(sub, test_mean, test_std))
		self.fig.canvas.draw()
		# plt.savefig("test.png")
 
	def collect_samples_vec(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):
		start_state = self.env.observe()
		samples = 0
		done = False
		states = []
		next_states = []
		actions = []
		# mean_actions = []
		rewards = []
		values = []
		q_values = []
		real_rewards = []
		log_probs = []
		dones = []
		noise = self.base_noise * self.explore_noise.value
		self.gpu_model.set_noise(noise)
 
		state = start_state
		total_reward1 = 0
		total_reward2 = 0
		calculate_done1 = False
		calculate_done2 = False
		self.total_rewards = []
		start = time.time()
		while samples < num_samples:
			kin_state = self.env.get_reference()
			state = self.env.observe()
			with torch.no_grad():
				action, mean_action = self.gpu_model.sample_actions(kin_state)
				log_prob = self.gpu_model.calculate_prob(kin_state, action, mean_action)
 
				states.append(kin_state.clone())
				actions.append(action.clone())
				log_probs.append(log_prob.clone())

				# self.phase += 1
				# self.phase %= 40
				# self.env.set_reference(np.concatenate((self.phase[:, None], self.reference[np.arange(self.num_envs), self.phase]),axis=1).astype(np.float32))
				# self.current_phase[:, 6:9] += action.cpu().numpy()[:, -6:-3] * 5
				dist, ind = tree_query(self.tree, self.current_phase, self.num_envs, self.feature_mean, self.feature_std, noise=action.cpu().numpy()[:, -9:])
				# ind = ind[np.arange(self.num_envs), np.random.randint(0, 1, self.num_envs)]
				reward = torch.zeros(400, device=device)
				for t in range(5):
					self.current_phase = self.phase[ind, t].copy()
					self.next_phase = self.phase[ind, t+5].copy()
					self.set_reference(ind, t=t)
					self.set_reference_velocity(ind, t=t)

					#update root reference
					self.current_root_pos += np.matmul(self.current_root_orientation, self.root_linear_vel[ind, t, :, None]).squeeze()
					self.current_root_orientation[:, :, :] = np.matmul(self.root_angular_vel[ind, t, :, :], self.current_root_orientation)
					# velocity = np.array([-50, 0, 0])
					# self.current_root_pos += np.matmul(self.current_root_orientation, np.repeat(velocity[np.newaxis,:], self.num_envs, axis=0)[:, :, None]).squeeze()

					self.reference[:, 0:3] = np.matmul(self.base_rot, (self.current_root_pos + self.store_data[ind, t, :])[:, :, None]).squeeze() / 100
					self.reference[:, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot, np.matmul(self.current_root_orientation, self.bone_orientation[ind, t, :]))).as_quat()[:,[3,0,1,2]]
					self.reference[self.reference[:, 3] < 0, 3:7] *= -1
					self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
					self.env.set_reference_velocity(self.reference_velocity.astype(np.float32))

					with torch.no_grad():
						dynamics_obs = self.env.observe()
						dynamics_obs[:, -36:] = 0
						dynamic_act = self.dynamic_policy.sample_best_actions(dynamics_obs)
						next_state, reward_temp, done, _ = self.env.step(dynamic_act)
						reward += reward_temp.clone()
				kin_state = self.env.get_reference()
	 
				dones.append(done.clone())
	 
				next_states.append(kin_state.clone())

			# reward = self.discriminator.compute_disc_reward(state[:, np.concatenate(([0], np.arange(5,73), np.arange(130, 136)))], next_state[:, np.concatenate(([0], np.arange(5,73), np.arange(130, 136)))]) * 0.0 + reward

				rewards.append(reward.clone() / 5)

				kin_state = kin_state.clone()

				#reset done state
				done_cpu = done.cpu().numpy()
				if (done_cpu.sum() > 0):
					self.current_phase[done_cpu==1, :] = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (done_cpu.sum(), 1))
					self.next_phase[done_cpu==1, :] = np.tile(self.phase[self.starting_frame, 5, 0:9].copy(), (done_cpu.sum(), 1))
					self.current_root_pos[done_cpu==1, :] = 0
					self.current_root_orientation[done_cpu==1, :, :] = 0
					self.current_root_orientation[done_cpu==1, 0, 0] = 1
					self.current_root_orientation[done_cpu==1, 1, 1] = 1
					self.current_root_orientation[done_cpu==1, 2, 2] = 1
					self.reference[done_cpu==1, 0:3] = np.matmul(self.base_rot[done_cpu==1], (self.current_root_pos[done_cpu==1] + self.store_data[np.ones(done_cpu.sum(), dtype=int) * self.starting_frame, 0, :])[:, :, None]).squeeze() / 100
					self.reference[done_cpu==1, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot[done_cpu==1], np.matmul(self.current_root_orientation[done_cpu==1], self.bone_orientation[np.ones(done_cpu.sum(), dtype=int) * self.starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
					self.set_reference(np.ones(done_cpu.sum(), dtype=int)*self.starting_frame, 0, done_cpu==1)
				self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
				self.env.done_reset(done_cpu.astype(bool))
	 
				samples += 1
	 
				time_limit_done = self.env.reset_time_limit()

				if (time_limit_done.sum() > 0):
					self.current_phase[time_limit_done==1, :] = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (time_limit_done.sum(), 1))
					self.next_phase[time_limit_done==1, :] = np.tile(self.phase[self.starting_frame, 5, 0:9].copy(), (time_limit_done.sum(), 1))
					self.current_root_pos[time_limit_done==1, :] = 0
					self.current_root_orientation[time_limit_done==1, :, :] = 0
					self.current_root_orientation[time_limit_done==1, 0, 0] = 1
					self.current_root_orientation[time_limit_done==1, 1, 1] = 1
					self.current_root_orientation[time_limit_done==1, 2, 2] = 1
					self.reference[time_limit_done==1, 0:3] = np.matmul(self.base_rot[time_limit_done==1], (self.current_root_pos[time_limit_done==1] + self.store_data[np.ones(time_limit_done.sum(), dtype=int) * self.starting_frame, 0, :])[:, :, None]).squeeze() / 100
					self.reference[time_limit_done==1, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot[time_limit_done==1], np.matmul(self.current_root_orientation[time_limit_done==1], self.bone_orientation[np.ones(time_limit_done.sum(), dtype=int) * self.starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
					self.set_reference(np.ones(time_limit_done.sum(), dtype=int)*self.starting_frame, 0, time_limit_done==1)
				self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
				self.env.done_reset(time_limit_done)		

		print("sim time", time.time() - start)
		start = time.time()
		counter = num_samples - 1
		R = self.gpu_model.get_value(kin_state)
		while counter >= 0:
			R = R * (1 - dones[counter].unsqueeze(-1))
			R = 0.99 * R + rewards[counter].unsqueeze(-1)
			q_values.insert(0, R)
			counter -= 1
			#print(len(q_values))
		for i in range(num_samples):
			self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)
		self.total_rewards = self.env.total_rewards.cpu().numpy().tolist()
		print("processing time", time.time() - start)

	def collect_samples_vec_dynamics(self, num_samples, start_state=None, noise=-2.5, env_index=0, random_seed=1):
		start_state = self.env.observe()
		samples = 0
		done = False
		states = []
		next_states = []
		actions = []
		rewards = []
		values = []
		q_values = []
		real_rewards = []
		log_probs = []
		dones = []
		time_limit_dones = []
		noise = self.base_noise * self.explore_noise.value
		self.gpu_model.set_noise(noise)
 
		state = start_state
		total_reward1 = 0
		total_reward2 = 0
		calculate_done1 = False
		calculate_done2 = False
		self.total_rewards = []
		start = time.time()
		while samples < num_samples:
			with torch.no_grad():
				kin_state = self.env.get_reference()
				# kinematic_action = self.kinematic_policy.sample_best_actions(kin_state)
 

				# dist, ind = tree_query(self.tree, self.current_phase, self.num_envs, self.feature_mean, self.feature_std, noise=kinematic_action.cpu().numpy()[:, -9:] * 0)

				# ind = ind[np.arange(self.num_envs), np.random.randint(0, 1, self.num_envs)]
				# ind = self.current_index
				for t in range(5):
					state = self.env.observe()
					action, mean_action = self.gpu_model.sample_actions(state)
					log_prob = self.gpu_model.calculate_prob(state, action, mean_action)
					states.append(state.clone())
					actions.append(action.clone())
					log_probs.append(log_prob.clone())
					self.current_phase = self.phase[self.current_index, 0].copy()
					self.next_phase = self.phase[self.current_index, 5:20:5].copy().reshape(self.current_index.shape[0], -1)
					self.set_reference(self.current_index, t=0)

					#update root reference
					for k in range(2):
						self.current_root_pos += np.matmul(self.current_root_orientation, self.root_linear_vel[self.current_index, k, :, None]).squeeze()
						self.current_root_orientation[:, :, :] = np.matmul(self.root_angular_vel[self.current_index, k, :, :], self.current_root_orientation)
						self.reference[:, 0:3] = np.matmul(self.base_rot, (self.current_root_pos + self.store_data[self.current_index, k, :])[:, :, None]).squeeze() / 100
						self.reference[:, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot, np.matmul(self.current_root_orientation, self.bone_orientation[self.current_index, k, :]))).as_quat()[:,[3,0,1,2]]
					self.reference[self.reference[:, 3] < 0, 3:7] *= -1
					self.env.set_reference(np.concatenate((self.current_phase, self.next_phase ,self.reference), axis=1).astype(np.float32))
					next_state, reward, done, _ = self.env.step(action)

					# print(reward[0], done[0])

					reward = reward #- 0.001 * torch.from_numpy(dist[np.arange(self.num_envs), np.random.randint(0, 1, 400)]).cuda()	 
					dones.append(done.clone())
		 
					next_states.append(next_state.clone())
					rewards.append(reward.clone())

					state = next_state.clone()
					self.current_index += 2

					#reset done state
					done_cpu = done.cpu().numpy()
					if (done_cpu.sum() > 0):
						#process count and starting index
						self.sampling_count[self.sampling_starting_index[done_cpu==1]] += self.env.num_steps[done_cpu==1]
						self.sampling_count.clip(1, 1000000)
						self.env.num_steps[done_cpu==1] = 0
						prob = 1.0 / self.sampling_count
						prob = prob / prob.sum() 
						starting_index = np.random.choice(self.sampling_count.shape[0], size=done_cpu.sum(), p=prob)
						self.sampling_starting_index[done_cpu==1] = starting_index
						starting_frame = self.starting_index_array[starting_index]

						#processed done and starting index for idle
						# self.idle_sampling_count[self.idle_sampling_starting_index[(done_cpu[400:]==1)]] += self.env.num_steps[400:][done_cpu[400:]==1]
						# self.idle_sampling_count.clip(1, 1000000)
						# self.env.num_steps[400:][done_cpu[400:]==1] = 0
						# prob = 1.0 / self.idle_sampling_count
						# prob = prob / prob.sum() 
						# idle_starting_index = np.random.choice(self.idle_sampling_count.shape[0], size=done_cpu[400:].sum(), p=prob)
						# self.idle_sampling_starting_index[(done_cpu[400:]==1)] = idle_starting_index
						# idle_starting_frame = self.idle_starting_index_array[idle_starting_index]

						# self.walking_sampling_count[self.walking_sampling_starting_index[done_cpu[0:400]==1]] += self.env.num_steps[0:400][done_cpu[0:400]==1]
						# self.walking_sampling_count.clip(1, 1000000)
						# self.env.num_steps[0:400][done_cpu[0:400]==1] = 0
						# prob = 1.0 / self.walking_sampling_count
						# prob = prob / prob.sum() 
						# walking_starting_index = np.random.choice(self.walking_sampling_count.shape[0], size=done_cpu[0:400].sum(), p=prob)
						# self.walking_sampling_starting_index[(done_cpu[0:400]==1)] = walking_starting_index
						# walking_starting_frame = self.walking_starting_index_array[walking_starting_index]

						# starting_frame = np.concatenate((walking_starting_frame, idle_starting_frame))
						

						self.current_phase[done_cpu==1, :] = self.phase[starting_frame, 0, 0:9].copy()
						self.next_phase[done_cpu==1, :] = self.phase[starting_frame, 5:20:5, 0:9].copy().reshape(done_cpu.sum(), -1)
						self.current_index[done_cpu==1] = starting_frame
						self.current_root_pos[done_cpu==1, :] = 0
						self.current_root_orientation[done_cpu==1, :, :] = 0
						self.current_root_orientation[done_cpu==1, 0, 0] = 1
						self.current_root_orientation[done_cpu==1, 1, 1] = 1
						self.current_root_orientation[done_cpu==1, 2, 2] = 1
						self.reference[done_cpu==1, 0:3] = np.matmul(self.base_rot[done_cpu==1], (self.current_root_pos[done_cpu==1] + self.store_data[starting_frame, 0, :])[:, :, None]).squeeze() / 100
						self.reference[done_cpu==1, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot[done_cpu==1], np.matmul(self.current_root_orientation[done_cpu==1], self.bone_orientation[starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
						self.set_reference(starting_frame, 0, done_cpu==1)
						self.set_reference_velocity(starting_frame, 0, done_cpu==1)
						self.reference_velocity[done_cpu==1, 0:3] = self.root_linear_vel[starting_frame, 0, :, None].squeeze() / 100.0 * 60.0
					self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
					self.env.set_reference_velocity(self.reference_velocity.astype(np.float32))
					self.env.done_reset(done_cpu.astype(bool))
		 
					samples += 1
		 
					time_limit_done = self.env.reset_time_limit()
					time_limit_dones.append(torch.from_numpy(time_limit_done.copy()).to(device))

					if (time_limit_done.sum() > 0):
						#process count and starting index
						self.sampling_count[self.sampling_starting_index[time_limit_done==1]] += self.env.num_steps[time_limit_done==1]
						self.sampling_count.clip(1, 1000000)
						self.env.num_steps[time_limit_done==1] = 0
						prob = 1.0 / self.sampling_count
						prob = prob / prob.sum() 
						starting_index = np.random.choice(self.sampling_count.shape[0], size=time_limit_done.sum(), p=prob)
						self.sampling_starting_index[time_limit_done==1] = starting_index
						starting_frame = self.starting_index_array[starting_index]
						
						# divide dataset
						# self.idle_sampling_count[self.idle_sampling_starting_index[(time_limit_done[400:]==1)]] += self.env.num_steps[400:][time_limit_done[400:]==1]
						# self.idle_sampling_count.clip(1, 1000000)
						# self.env.num_steps[400:][time_limit_done[400:]==1] = 0
						# prob = 1.0 / self.idle_sampling_count
						# prob = prob / prob.sum() 
						# idle_starting_index = np.random.choice(self.idle_sampling_count.shape[0], size=time_limit_done[400:].sum(), p=prob)
						# self.idle_sampling_starting_index[(time_limit_done[400:]==1)] = idle_starting_index
						# idle_starting_frame = self.idle_starting_index_array[idle_starting_index]

						# self.walking_sampling_count[self.walking_sampling_starting_index[time_limit_done[0:400]==1]] += self.env.num_steps[0:400][time_limit_done[0:400]==1]
						# self.walking_sampling_count.clip(1, 1000000)
						# self.env.num_steps[0:400][time_limit_done[0:400]==1] = 0
						# prob = 1.0 / self.walking_sampling_count
						# prob = prob / prob.sum() 
						# walking_starting_index = np.random.choice(self.walking_sampling_count.shape[0], size=time_limit_done[0:400].sum(), p=prob)
						# self.walking_sampling_starting_index[(time_limit_done[0:400]==1)] = walking_starting_index
						# walking_starting_frame = self.walking_starting_index_array[walking_starting_index]

						# starting_frame = np.concatenate((walking_starting_frame, idle_starting_frame))
						

						self.current_phase[time_limit_done==1, :] = self.phase[starting_frame, 0, 0:9].copy()
						self.next_phase[time_limit_done==1, :] = self.phase[starting_frame, 5:20:5, 0:9].copy().reshape(time_limit_done.sum(), -1)
						self.current_index[time_limit_done==1] = starting_frame
						self.current_root_pos[time_limit_done==1, :] = 0
						self.current_root_orientation[time_limit_done==1, :, :] = 0
						self.current_root_orientation[time_limit_done==1, 0, 0] = 1
						self.current_root_orientation[time_limit_done==1, 1, 1] = 1
						self.current_root_orientation[time_limit_done==1, 2, 2] = 1
						self.reference[time_limit_done==1, 0:3] = np.matmul(self.base_rot[time_limit_done==1], (self.current_root_pos[time_limit_done==1] + self.store_data[starting_frame, 0, :])[:, :, None]).squeeze() / 100
						self.reference[time_limit_done==1, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot[time_limit_done==1], np.matmul(self.current_root_orientation[time_limit_done==1], self.bone_orientation[starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
						self.set_reference(starting_frame, 0, time_limit_done==1)
						self.set_reference_velocity(starting_frame, 0, time_limit_done==1)
						self.reference_velocity[time_limit_done==1, 0:3] = self.root_linear_vel[starting_frame, 0, :, None].squeeze() / 100.0 * 60.0
					
					self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
					self.env.set_reference_velocity(self.reference_velocity.astype(np.float32))
					self.env.done_reset(time_limit_done)		
				# self.current_index += 5		

		print("sim time", time.time() - start)
		start = time.time()
		counter = num_samples - 1
		R = self.gpu_model.get_value(state)
		while counter >= 0:
			R = self.gpu_model.get_value(next_states[counter]) * time_limit_dones[counter].unsqueeze(-1).float() + (1 - time_limit_dones[counter].unsqueeze(-1).float()) * R
			R = R * (1 - dones[counter].unsqueeze(-1))
			R = 0.99 * R + rewards[counter].unsqueeze(-1)
			q_values.insert(0, R)
			counter -= 1
			#print(len(q_values))
		for i in range(num_samples):
			self.storage.push(states[i], actions[i], next_states[i], rewards[i], q_values[i], log_probs[i], self.num_envs)
		self.total_rewards = self.env.total_rewards.cpu().numpy().tolist()
		print("processing time", time.time() - start)

	def update_discriminator(self, batch_size, num_epoch):
		self.discriminator.train()
		optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
		for k in range(num_epoch):
			batch_states, batch_next_states = self.storage.discriminator_sample(batch_size)
			policy_d = self.discriminator.compute_disc(batch_states[:, np.concatenate(([0], np.arange(5,73), np.arange(130, 136)))], batch_next_states[:, np.concatenate(([0], np.arange(5,73), np.arange(130, 136)))])
			policy_loss = (policy_d + torch.ones(policy_d.size(), device=device))**2
			policy_loss = policy_loss.mean()

			idx = np.random.choice(10000, batch_size)
			batch_expert_states, batch_expert_next_states = self.sample_motion_data(idx)
			expert_d = self.discriminator.compute_disc(batch_expert_states, batch_expert_next_states)
			expert_loss = (expert_d - torch.ones(expert_d.size(), device=device))**2
			expert_loss = expert_loss.mean()

			grad_penalty = self.discriminator.grad_penalty(batch_expert_states, batch_expert_next_states)

			print(policy_loss, expert_loss, grad_penalty)

			total_loss = policy_loss + expert_loss + 5 * grad_penalty
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

 
	def update_critic(self, batch_size, num_epoch):
		self.gpu_model.train()
		optimizer = self.critic_optimizer#optim.Adam(self.gpu_model.parameters(), lr=10*self.lr)

		storage = self.storage
		gpu_model = self.gpu_model

		for k in range(num_epoch):
			batch_states, batch_q_values = storage.critic_sample(batch_size)
			# batch_q_values = batch_q_values# / self.max_reward.value
			v_pred = gpu_model.get_value(batch_states)
			
			loss_value = (v_pred - batch_q_values)**2
			loss_value = 0.5 * loss_value.mean()

			optimizer.zero_grad()
			loss_value.backward()
			nn.utils.clip_grad_norm_(self.gpu_model.parameters(), 1.0)
			optimizer.step()
			
 
	def update_actor(self, batch_size, num_epoch):
		self.gpu_model.train()
		optimizer = self.actor_optimizer#optim.Adam(self.gpu_model.parameters(), lr=self.lr)

		storage = self.storage
		gpu_model = self.gpu_model
		model_old = self.model_old
		params_clip = self.params.clip
 
		for k in range(num_epoch):
			batch_states, batch_actions, batch_next_states, batch_q_values, batch_log_probs = storage.actor_sample(batch_size)
 
			# batch_q_values = batch_q_values# / self.max_reward.value
 
			with torch.no_grad():
				v_pred_old = gpu_model.get_value(batch_states)

			batch_advantages = (batch_q_values - v_pred_old)
 
			probs, mean_actions = gpu_model.calculate_prob_gpu(batch_states, batch_actions)
			probs_old = batch_log_probs#model_old.calculate_prob_gpu(batch_states, batch_actions)
			ratio = (probs - (probs_old)).exp()
			ratio = ratio.unsqueeze(1)
			surr1 = ratio * batch_advantages
			surr2 = ratio.clamp(1-params_clip, 1+params_clip) * batch_advantages
			loss_clip = -(torch.min(surr1, surr2)).mean()

			#regularize action difference temporally: https://arxiv.org/pdf/2012.06644.pdf
			# loss_T = ((gpu_model.sample_best_actions(batch_states)-gpu_model.sample_best_actions(batch_next_states))**2).mean()

			#regularize action difference for close state
			# with torch.no_grad():
			# 	noisy_batch_states = batch_states + torch.randn(batch_states.size(), device=device) * 0.01
			# loss_S = ((gpu_model.sample_best_actions(batch_states)-gpu_model.sample_best_actions(noisy_batch_states))**2).mean()
			#loss_gate = gpu_model.evaluate_policy_gate_l2(batch_states)
			total_loss = loss_clip + 0.01 * (mean_actions**2).mean()# + loss_gate# + 0.5 * loss_T + 0.5 * loss_S
			
			optimizer.zero_grad()
			total_loss.backward()
			nn.utils.clip_grad_norm_(self.gpu_model.parameters(), 1.0)
			optimizer.step()
		#print(self.shared_obs_stats.mean.data)
		if self.lr > 1e-4:
			self.lr *= 0.99
		else:
			self.lr = 1e-4
 
	def save_model(self, filename):
		torch.save(self.gpu_model.state_dict(), filename)
 
	def save_shared_obs_stas(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self.shared_obs_stats, output, pickle.HIGHEST_PROTOCOL)
 
	def save_statistics(self, filename):
		statistics = [self.time_passed, self.num_samples, self.test_mean, self.test_std, self.noisy_test_mean, self.noisy_test_std]
		with open(filename, 'wb') as output:
			pickle.dump(statistics, output, pickle.HIGHEST_PROTOCOL)

	def sample_motion_data(self, index):
		base_rot_quat = np.zeros((index.shape[0], 4))
		base_rot_quat[:, 0] = 0.707
		base_rot_quat[:, 3] = 0.707
		base_rot = Rotation.from_quat(base_rot_quat).as_matrix()

		motion = np.zeros((index.shape[0], 75), dtype=np.float32)
		motion[:, 0:3] = np.matmul(base_rot, (np.zeros((index.shape[0], 3)) + self.store_data[index, 0, :])[:, :, None]).squeeze() / 100
		motion[:, 4] = 1.0
		motion[:, 7:11] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0, 19, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 11:15] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0, 20, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 15:19] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  21, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 19:23] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  15, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 23:27] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  16, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 27:31] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  17, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 31:35] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  1, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 35:39] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  2, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 39:43] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  3, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 43:47] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  4, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 47:51] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  5, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 51:55] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  11, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		motion[:, 55:59] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  12, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 59:63] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  13, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 63:67] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  7, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		motion[:, 67:71] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  8, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		motion[:, 71:75] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 0,  9, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		phase = self.phase[index, 0].copy()

		next_motion = np.zeros((index.shape[0], 75), dtype=np.float32)
		motion[:, 0:3] = np.matmul(base_rot, (np.zeros((index.shape[0], 3)) + self.store_data[index, 1, :])[:, :, None]).squeeze() / 100
		next_motion[:, 4] = 1.0
		next_motion[:, 7:11] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1, 19, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 11:15] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1, 20, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 15:19] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  21, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 19:23] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  15, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 23:27] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  16, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 27:31] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  17, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 31:35] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  1, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 35:39] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  2, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 39:43] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  3, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 43:47] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  4, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 47:51] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  5, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 51:55] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  11, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		next_motion[:, 55:59] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  12, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 59:63] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  13, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 63:67] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  7, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		next_motion[:, 67:71] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  8, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_motion[:, 71:75] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, 1,  9, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		next_phase = self.phase[index, 1].copy()
		return torch.from_numpy(np.concatenate((motion[:, np.concatenate(([2], np.arange(7,75)))], phase[:, 0:6]),axis=1).astype(np.float32)).to(device), torch.from_numpy(np.concatenate((next_motion[:, np.concatenate(([2], np.arange(7,75)))], next_phase[:, 0:6]), axis=1).astype(np.float32)).to(device)
 
	def collect_samples_multithread(self):
		#queue = Queue.Queue()
		import time
		self.start = time.time()
		self.lr = 1e-3
		self.weight = 10
		num_threads = 1
		self.num_samples = 0
		self.time_passed = 0
		score_counter = 0
		total_thread = 0
		max_samples = self.num_envs * 50
		self.storage = PPOStorage(self.num_inputs, self.num_outputs, max_size=max_samples)
		seeds = [
			i * 100 for i in range(num_threads)
		]
		self.explore_noise = mp.Value("f", -2.5)
		if self.train_dynamics:
			self.base_noise = np.ones(self.num_outputs)
		else:
			self.base_noise = np.ones(9)
		noise = self.base_noise * self.explore_noise.value
		self.model.set_noise(noise)
		self.gpu_model.set_noise(noise)
		# import ipdb; ipdb.set_trace()
		self.env.set_reference(np.concatenate((self.current_phase, self.next_phase, self.reference), axis=1).astype(np.float32))
		self.env.reset()
		for iterations in range(200000):
			iteration_start = time.time()
			print(self.model_name)
			while self.storage.counter < max_samples:
				if self.train_dynamics:
					self.collect_samples_vec_dynamics(50, noise=noise)
				else:
					self.collect_samples_vec(100, noise=noise)
			start = time.time()

			self.update_critic(max_samples//4, 20)
			self.update_actor(max_samples//4, 20)
			# self.update_discriminator(max_samples//4, 40)
			self.storage.clear()
 
			if (iterations+1) % 100 == 0:
				self.run_test_with_noise(num_test=2)
				self.plot_statistics()
				plt.savefig(self.model_name+"test.png")

			print("update policy time", time.time()-start)
			print("iteration time", iterations, time.time()-iteration_start)
 
			if (iterations+0) % 1000 == 999:
				self.save_model(self.model_name+"iter%d.pt"%(iterations))
 
		self.save_reward_stats("reward_stats.npy")
		self.save_model(self.model_name+"final.pt")
 
	def add_env(self, env):
		self.env_list.append(env)

	def load_reference(self):
		self.tree, self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.feature_mean, self.feature_std, self.bone_velocity = build_KD_tree()


		# motion, next_motion = self.sample_motion_data(np.random.choice(1000, 10000))
		# motion_mean = np.mean(motion.cpu().numpy(), axis=0)
		# motion_std = np.mean(motion.cpu().numpy(), axis=0)
		# self.discriminator.set_normalizer(motion_mean, motion_std)
		
		self.current_index = np.ones(self.num_envs, dtype=np.int) * self.starting_frame
		self.current_phase = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (self.num_envs, 1))
		self.next_phase = np.tile(self.phase[self.starting_frame, 5:20:5, 0:9].copy().flatten(), (self.num_envs, 1))
		print(self.next_phase.shape)
		self.current_root_pos = np.zeros((self.num_envs, 3))
		self.current_root_orientation = np.zeros((self.num_envs, 3, 3))
		self.current_root_orientation[:, 0, 0] = 1
		self.current_root_orientation[:, 1, 1] = 1
		self.current_root_orientation[:, 2, 2] = 1

		base_rot_quat = np.zeros((self.num_envs, 4))
		base_rot_quat[:, 0] = 0.707
		base_rot_quat[:, 3] = 0.707
		self.base_rot = Rotation.from_quat(base_rot_quat).as_matrix()

		self.reference = np.zeros((self.num_envs, 75))
		self.reference_velocity = np.zeros((self.num_envs, 57))

		self.reference[:, 0:3] = np.matmul(self.base_rot, (self.current_root_pos + self.store_data[np.ones(self.num_envs, dtype=np.int) * self.starting_frame, 0, :])[:, :, None]).squeeze() / 100
		self.reference[:, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot, np.matmul(self.current_root_orientation, self.bone_orientation[np.ones(self.num_envs, dtype=np.int) * self.starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
		self.set_reference(np.ones(self.num_envs, dtype=np.int)*self.starting_frame)
		self.set_reference_velocity(np.ones(self.num_envs, dtype=np.int)*self.starting_frame)
		self.env.set_reference(np.concatenate((self.current_phase, self.next_phase ,self.reference), axis=1).astype(np.float32))
		self.env.set_reference_velocity(self.reference_velocity.astype(np.float32))

		dataset_size = [3786, 4432, 4432, 3786, 4005, 4005, 3691, 3691, 11999, 11999, 3687, 3687, 4049, 4049, 3666, 3666, 4181, 4181, 3045, 3045]
		# dataset_size = [3786, 3786, 3045, 3045]
		# self.starting_index_array = np.arange(2043) + 2
		self.starting_index_array = np.arange(2784) + 2
		processed_data_size = dataset_size[0]
		for i in range(1, len(dataset_size)):
			self.starting_index_array = np.concatenate((np.arange(dataset_size[i]-1002)+2+processed_data_size, self.starting_index_array))
			processed_data_size += dataset_size[i]
		print(self.starting_index_array.shape)
		self.sampling_starting_index = np.zeros(self.num_envs, np.int32)
		self.sampling_count = np.ones(self.starting_index_array.shape, np.int32)

		#divide dataset
		# dataset_size = [3786, 3786]
		# self.walking_starting_index_array = np.arange(2784) + 2
		# processed_data_size = dataset_size[0]
		# for i in range(1, len(dataset_size)):
		# 	self.walking_starting_index_array = np.concatenate((np.arange(dataset_size[i]-1002)+2+processed_data_size, self.walking_starting_index_array))
		# 	processed_data_size += dataset_size[i]

		# dataset_size = [3045, 3045]
		# self.idle_starting_index_array = np.arange(2043) + 2 + 3786 + 3786
		# processed_data_size = dataset_size[0] + 3786 + 3786
		# for i in range(1, len(dataset_size)):
		# 	self.idle_starting_index_array = np.concatenate((np.arange(dataset_size[i]-1002)+2+processed_data_size, self.idle_starting_index_array))
		# 	processed_data_size += dataset_size[i]

		# self.idle_sampling_starting_index = np.zeros(self.num_envs//2, np.int32)
		# self.idle_sampling_count = np.ones(self.idle_starting_index_array.shape, np.int32)

		# self.walking_sampling_starting_index = np.zeros(self.num_envs//2, np.int32)
		# self.walking_sampling_count = np.ones(self.walking_starting_index_array.shape, np.int32)

	def set_reference(self, index, t=0, env_id=None):
		if env_id is None:
			env_id = np.arange(self.num_envs)
		#set left leg
		self.reference[env_id, 7:11] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 19, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 11:15] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 20, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 15:19] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 21, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 19:23] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 15, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 23:27] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 16, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 27:31] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 17, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 31:35] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 1, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 35:39] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 2, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 39:43] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 3, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 43:47] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 4, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 47:51] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 5, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 51:55] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 11, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		self.reference[env_id, 55:59] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 12, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 59:63] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 13, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 63:67] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 7, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		self.reference[env_id, 67:71] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 8, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[env_id, 71:75] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 9, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])

		# self.reference[env_id, 63:75] = np.array([9.98472234e-01,  4.41946509e-03,  5.41092287e-02, -1.02887418e-02,  8.80818265e-01, -1.23713898e-01, 1.37720908e-01,  4.35760264e-01,  8.52879586e-01, -1.97142974e-01,  3.55116974e-01,  3.28059438e-01])		
		# self.reference[env_id, 51:63] = np.array([9.98673887e-01,  2.82588986e-02, -3.39619410e-02, 2.64289517e-02,  8.85221521e-01, -1.26977037e-01, -7.39741470e-02, -4.41347387e-01,  7.71478156e-01, -2.40705177e-01, -3.74336623e-01, -4.54702722e-01])


		# self.reference[env_id, 63:75] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, 0.13, 0.438, 0.85, -0.2, 0.35, 0.34])
		# self.reference[env_id, 51:63] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, -0.13, -0.438, 0.85, -0.2, -0.35, -0.34])

	def set_reference_velocity(self, index, t=0, env_id=None):
		if env_id is None:
			env_id = np.arange(self.num_envs)
		#set left leg
		self.reference_velocity[env_id, 6:9] = self.bone_velocity[index, t, 19, :]
		self.reference_velocity[env_id, 9:12] = self.bone_velocity[index, t, 20, :]
		self.reference_velocity[env_id, 12:15] = self.bone_velocity[index, t, 21, :]
		self.reference_velocity[env_id, 15:18] = self.bone_velocity[index, t, 15, :]
		self.reference_velocity[env_id, 18:21] = self.bone_velocity[index, t, 16, :]
		self.reference_velocity[env_id, 21:24] = self.bone_velocity[index, t, 17, :]
		self.reference_velocity[env_id, 24:27] = self.bone_velocity[index, t, 1, :]
		self.reference_velocity[env_id, 27:30] =self.bone_velocity[index, t, 2, :]
		self.reference_velocity[env_id, 30:33] = self.bone_velocity[index, t, 3, :]
		self.reference_velocity[env_id, 33:36] = self.bone_velocity[index, t, 4, :]
		self.reference_velocity[env_id, 36:39] = self.bone_velocity[index, t, 5, :]
		self.reference_velocity[env_id, 39:42] = self.bone_velocity[index, t, 11, :]
		self.reference_velocity[env_id, 42:45] = self.bone_velocity[index, t, 12, :]
		self.reference_velocity[env_id, 45:48] = self.bone_velocity[index, t, 13, :]
		self.reference_velocity[env_id, 48:51] = self.bone_velocity[index, t, 7, :]
		self.reference_velocity[env_id, 51:54] = self.bone_velocity[index, t, 8, :]
		self.reference_velocity[env_id, 54:57] = self.bone_velocity[index, t, 9, :]


 
def mkdir(base, name):
	path = os.path.join(base, name)
	if not os.path.exists(path):
		os.makedirs(path)
	return path
 
if __name__ == '__main__':
	import json
	from ruamel.yaml import YAML, dump, RoundTripDumper
	from raisimGymTorch.env.bin import motion_matching_3
	from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
	from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
 
	import torch
	import torch.optim as optim
	import torch.multiprocessing as mp
	import torch.nn as nn
	import torch.nn.functional as F
	from torch.autograd import Variable
	import torch.utils.data
	from model import ActorCriticNetMann, Discriminator, ActorCriticNet
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
	seed = 1#8
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.set_num_threads(1)
 
	# directories
	task_path = os.path.dirname(os.path.realpath(__file__))
	home_path = task_path + "/../../../../.."

	# config
	cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

	# create environment from the configuration file
	env = VecEnv(motion_matching_3.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
	print("env_created")
	# env.setTask()
	ppo = RL(env, [128, 128])  #128 128 for MANN
	ppo.num_envs = 800
	ppo.starting_frame = 200#10977 #400
	ppo.load_reference()
 
	ppo.base_dim = ppo.num_inputs
 
	ppo.model_name = "stats/release_test/"
	ppo.max_reward.value = 1#50
	# ppo.kinematic_policy.load_state_dict(torch.load("back_and_forth.pt"))
	# ppo.gpu_model.load_state_dict(torch.load("stats/strong_ankle/iter7999.pt"))
 
	#ppo.save_model(ppo.model_name)
	training_start = time.time()
	ppo.collect_samples_multithread()

	print("training time", time.time()-training_start)
 

