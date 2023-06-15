import json
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import motion_matching_2
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher

from process_mocap import KD_tree_Env, tree_query, coordinate_transform
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from model import ActorCriticNet, ActorCriticNetMann
import os
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_quaternion, axis_angle_to_matrix
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process_pose_output(pose):
   length = pose.shape[1]
   joint_output = torch.zeros((length, 4 * 17))
   urdf_to_bvh = [1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 15, 13, 16, 18, 14, 17, 19]
   for t in range(length):
   	for i in range(17):
   		data = pose[:, t, urdf_to_bvh[i], :]
   		quaternion = axis_angle_to_quaternion(data)[0, :]
   		joint_output[t, i * 4: i * 4 + 4] = quaternion.clone()

   root_linear_output = torch.zeros((length, 3))
   for i in range(length):
   	root_linear_output[i, :] = pose[0, i, 22, :]
   # root_linear_output[:, 2] *= -1

   root_angular_output = torch.zeros((length, 3, 3))
   for i in range(length):
   	root_angular_output[i, : ,:] = axis_angle_to_matrix(pose[0, i, 23, :])

   bone_orientation_output = torch.zeros((length, 3, 3))
   for i in range(length):
   	bone_orientation_output[i, :, :] = axis_angle_to_matrix(pose[0, i, 0, :])

   joint_output = joint_output.cpu().numpy()
   root_linear_output = root_linear_output.cpu().numpy()
   root_angular_output = root_angular_output.cpu().numpy()
   bone_orientation_output = bone_orientation_output.cpu().numpy()

   return joint_output, root_linear_output, root_angular_output, bone_orientation_output

task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
env = VecEnv(motion_matching_2.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
num_envs = 1
kd_tree_env = KD_tree_Env(num_envs)

picking_model = ActorCriticNetMann(233+4, env.action_space.shape[0], [128, 128])
picking_model.load_state_dict(torch.load("pick_and_place_model.pt")) 


picking_model.cuda()

carry_model = ActorCriticNetMann(226+4, env.action_space.shape[0], [128, 128])
carry_model.load_state_dict(torch.load("carry_model.pt"))
carry_model.cuda()

walk_model = ActorCriticNetMann(226+4, env.action_space.shape[0], [128, 128])
walk_model.load_state_dict(torch.load("walk_model.pt"))
walk_model.cuda()


#load walking data
pose_first_walk = torch.load("side_step_left.pt")
pose_second_walk = torch.load("side_step_right.pt")
pose_third_walk = torch.load("turn_left_and_walk.pt")
pose_fourth_walk = torch.load("turn_right_and_walk.pt")

current_phase = np.zeros((1, 9))
next_phase = np.zeros((1, 27))
base_rot_quat = np.zeros((1, 4))
base_rot_quat[:, 0] = 0.707
base_rot_quat[:, 3] = 0.707
base_rot = Rotation.from_quat(base_rot_quat).as_matrix()
root_offset = np.zeros((1, 3))
root_offset[0, 1] = 1.

current_phase = np.zeros((1, 9))
next_phase = np.zeros((1, 27))
base_rot_quat = np.zeros((1, 4))
base_rot_quat[:, 0] = 0.707
base_rot_quat[:, 3] = 0.707
base_rot = Rotation.from_quat(base_rot_quat).as_matrix()
root_offset = np.zeros((1, 3))
root_offset[0, 1] = 1.

current_root_pos = np.zeros((1, 3))
current_root_pos[0, 0] = 0
current_root_orientation = np.zeros((1, 3, 3))
current_root_orientation[:, 0, 0] = 1
current_root_orientation[:, 1, 1] = 1
current_root_orientation[:, 2, 2] = 1

identity_orientation = np.zeros((1, 3, 3))
identity_orientation[:, 0, 0] = 1
identity_orientation[:, 1, 1] = 1
identity_orientation[:, 2, 2] = 1

def pick_up(reset=False, start_time=320, timer=240, debug=False, debug_timing=0, timestep=0.016):
	global current_root_pos, current_root_orientation, counter, obs, save_state
	print(reset)
	kd_tree_env.current_root_pos = current_root_pos
	kd_tree_env.current_root_orientation[:, :, :] = current_root_orientation
	ind = np.zeros(num_envs, dtype=np.int) + start_time  # starting frame is 320
	for t in range(timer):
		kd_tree_env.set_reference(ind, 0)
		kd_tree_env.current_root_pos += np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.root_linear_vel[ind, 0, :, None]).squeeze()
		kd_tree_env.current_root_orientation[:, :, :] = np.matmul(identity_orientation, kd_tree_env.current_root_orientation)
		# print("root orientation ", kd_tree_env.current_root_orientation[0, : ,:])
		kd_tree_env.reference[:, 0:3] = np.matmul(kd_tree_env.base_rot, (kd_tree_env.current_root_pos * 100 + kd_tree_env.store_data[ind, 0, :])[:, :, None]).squeeze() / 100
		kd_tree_env.reference[:, 3:7] = Rotation.from_matrix(np.matmul(identity_orientation, np.matmul(kd_tree_env.current_root_orientation, identity_orientation))).as_quat()[:,[3,0,1,2]]
		kd_tree_env.reference[kd_tree_env.reference[:, 3] < 0, 3:7] *= -1
		env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))

		if t == 0 and reset:
			print("reset happens")
			env.reset()
			obs = env.observe()

		obs[0, 0:2] = 0
		obs[0, 10:10+68] = 0
		dynamic_act = picking_model.sample_best_actions(obs)
		if debug and t >= debug_timing:
			import ipdb; ipdb.set_trace()
		obs, rew, done, _ = env.step(dynamic_act)
		ind += 1
		import time; time.sleep(timestep)
		current_root_pos = kd_tree_env.current_root_pos
		current_root_orientation = kd_tree_env.current_root_orientation
		save_state[counter, :] = env.get_state()[0, :].cpu().numpy()
		counter += 1

def put_down(reset=False, start_time=320, timer=240, debug=False, debug_timing=0, timestep=0.016):
	global current_root_pos, current_root_orientation, counter, obs, save_state
	kd_tree_env.current_root_pos = current_root_pos
	kd_tree_env.current_root_orientation[:, :, :] = current_root_orientation
	ind = np.zeros(num_envs, dtype=np.int) + start_time  # starting frame is 320
	for t in range(timer):
		kd_tree_env.set_reference(ind, 0)
		kd_tree_env.current_root_pos += np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.root_linear_vel[ind, 0, :, None]).squeeze()
		kd_tree_env.current_root_orientation[:, :, :] = np.matmul(identity_orientation, kd_tree_env.current_root_orientation)
		# print("root orientation ", kd_tree_env.current_root_orientation[0, : ,:])
		kd_tree_env.reference[:, 0:3] = np.matmul(kd_tree_env.base_rot, (kd_tree_env.current_root_pos * 100 + kd_tree_env.store_data[ind, 0, :])[:, :, None]).squeeze() / 100
		kd_tree_env.reference[:, 3:7] = Rotation.from_matrix(np.matmul(identity_orientation, np.matmul(kd_tree_env.current_root_orientation, identity_orientation))).as_quat()[:,[3,0,1,2]]
		kd_tree_env.reference[kd_tree_env.reference[:, 3] < 0, 3:7] *= -1
		env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
		obs[0, 0:2] = 0
		dynamic_act = picking_model.sample_best_actions(obs)
		obs, rew, done, _ = env.step(dynamic_act)
		ind += 1
		import time; time.sleep(timestep)
		if debug and t >= debug_timing:
			import ipdb; ipdb.set_trace()
		current_root_pos = kd_tree_env.current_root_pos
		current_root_orientation = kd_tree_env.current_root_orientation
		save_state[counter, :] = env.get_state()[0, :].cpu().numpy()
		counter += 1

def walk(data, time_limit, reset=False, start_time=320, timer=240, debug=False, box_height=0.3, carry=False, debug_timing=0, timestep=0.016, box_width=0.3, warming_up_time=0, plot_torque=False, tuning_para=[0, 0]):
	global current_root_pos, current_root_orientation, counter, obs, save_state

	joint, linear_vel, angular_vel, bone_orientation = process_pose_output(data)
	max_time = joint.shape[0]
	starting_orientation = current_root_orientation.copy()
	starting_position = current_root_pos.copy()

	torque_profile = np.zeros(time_limit)
	torque_counter = 0

	for k in range(time_limit):
		if k < warming_up_time:
			t = 0
		elif k < max_time+warming_up_time:
			t = k - warming_up_time
		else:
			t = max_time - 1


		reference = np.zeros((1, 75))
		reference[0, 2] = 1
		reference[0, 3:7] = 0.5

		# set joint poses
		for i in range(17):
			reference[0, 7 + 4 * i: 11 + 4 * i] = joint[t][4*i:4*i+4].copy()

		if carry:
			reference[:, 63:75] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, 0.13, 0.438, 0.85, -0.2, 0.35, 0.34])
			reference[:, 51:63] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, -0.13, -0.438, 0.85, -0.2, -0.35, -0.34])
		if k >= max_time+warming_up_time and carry:
			reference[:, 7+4*6:7+4*7] = np.array([0.99, 0.141, 0, 0])
			# reference[:, 7+4*7:7+4*8] = np.array([0.99, 0.141, 0, 0])
			for j in range(6):
				reference[:,7+j*4:11+j*4] = np.array([1, 0, 0, 0])
				# reference[:, 7:7+4*1] = np.array([1, 0., 0, 0])
				# reference[:, 7+4*3:7+4*4] = np.array([1, 0., 0, 0])
			if carry:
				reference[:, 7:7+4*1] = np.array([0.99, 0.141, 0, 0])
				reference[:, 7+4*3:7+4*4] = np.array([0.99, 0.141, 0, 0])
			else:
				reference[:, 7:7+4*1] = np.array([1, 0., 0, 0])
				reference[:, 7+4*3:7+4*4] = np.array([1, 0., 0, 0])

			reference[:, 0:-20] = np.array([[ 8.59516173e-04, -5.81254891e-03,  9.78642710e-01,
						         5.57094933e-01,  4.40645942e-01,  4.42065507e-01,
						         5.47772286e-01,  9.91206066e-01,  1.30009731e-01,
						         2.45914852e-02,  1.80661908e-03,  9.99833341e-01,
						        -1.55529508e-02, -5.68300316e-03, -7.68756415e-03,
						         9.98632604e-01, -3.62268851e-02,  3.73101745e-02,
						         5.33728404e-03,  9.85719044e-01,  1.59653993e-01,
						        -5.09654667e-02, -1.64647958e-02,  9.99890601e-01,
						        -1.38006428e-02,  4.14220297e-03, -3.34209734e-03,
						         9.96033034e-01, -7.89337329e-03, -8.75871384e-02,
						         1.35787500e-02,  9.97240968e-01,  7.18584606e-02,
						        -9.70317292e-04, -1.85976432e-02,  9.99454805e-01,
						         3.19606880e-02, -4.31569488e-04, -8.27172170e-03,
						         9.99454805e-01,  3.19606880e-02, -4.31569209e-04,
						        -8.27173043e-03,  9.99693315e-01,  2.39724229e-02,
						        -3.23703558e-04, -6.20428566e-03,  9.89598261e-01,
						         1.39035230e-01,  3.69252512e-02,  1.00601603e-03,
						         9.98768461e-01, -2.34362139e-02, -3.64359179e-02,
						        -2.41811591e-02]])
		

		current_root_pos[0, 0:3]  = starting_position + np.matmul(starting_orientation[0, :, :], linear_vel[t, :, None].copy()).squeeze()
		current_root_orientation[0, :, :] = np.matmul(angular_vel[t, :, :], starting_orientation[0, :, :])
		reference[0, 0:3] = np.matmul(base_rot, (current_root_pos + root_offset)[:, :, None]).squeeze()
		reference[0, 2] = 0.9
		reference[0, 3:7] = Rotation.from_matrix(np.matmul(base_rot, np.matmul(current_root_orientation, bone_orientation[t:t+1, :, :]))).as_quat()[:,[3,0,1,2]]
		# reference[0, 0] = 0.016 * k
		env.set_reference(np.concatenate((current_phase, next_phase, reference), axis=1).astype(np.float32))

		if k == 0 and reset:
			print("reset happens")
			env.reset()
			obs = env.observe()

		# query controller
		index_offset = 4

		carry_obs = torch.zeros((1, 226+index_offset), device=device)
		carry_obs[:, 0:(10+136+57+index_offset)] = obs[:, 0:(10+136+57+index_offset)]
		carry_obs[:, 203+index_offset:210+index_offset] = obs[:, 211+index_offset:218+index_offset]
		carry_obs[:, 210+index_offset] = box_width  #hard code box width for now
		carry_obs[:, 211+index_offset] = box_height 
		carry_obs[:, 212+index_offset:219+index_offset] = obs[:, 219+index_offset:226+index_offset]

		carry_obs[:, 219+index_offset:226+index_offset] = obs[:, 226+index_offset:233+index_offset]
		if (k >= max_time+warming_up_time-75) and not carry:
			carry_obs[:, 0:2] = 0

		if not carry:
			carry_obs[:, -23:] = 0

		carry_obs[0, 0:2] /= 10
		carry_obs[0, 1] += tuning_para[0]
		carry_obs[0, 0] += tuning_para[1]

		if carry:
			# carry_obs[0, 0] /= 10
			# carry_obs[0, 1] /= 10
			# if k > warming_up_time + max_time:
			# 	carry_obs[0, 0:2] = 0
			dynamic_act = carry_model.sample_best_actions(carry_obs)
			if k < warming_up_time:
				obs[0, 0] = 0.0
				obs[0, 1] = 0.0
				obs[0, 10:10+68] = 0.0
				# obs[0,-30:] = 0.0
				obs[0, 209+index_offset] = np.sin(1.8 * np.pi)
				obs[0, 210+index_offset] = np.cos(1.8 * np.pi)
				obs[0, 205] = 0
				dynamic_act = (1 - k * 1.0 / warming_up_time) * picking_model.sample_best_actions(obs) + (k * 1.0 / warming_up_time) * dynamic_act
		else:
			dynamic_act = walk_model.sample_best_actions(carry_obs)

		obs, rew, done, _ = env.step(dynamic_act)

		torque_profile[torque_counter] = dynamic_act[0, 6].cpu().numpy()
		torque_counter += 1

		# import ipdb; ipdb.set_trace()
		import time; time.sleep(timestep)
		if debug and k >= debug_timing:
			import ipdb; ipdb.set_trace()
		save_state[counter, :] = env.get_state()[0, :].cpu().numpy()
		counter += 1
	# import ipdb; ipdb.set_trace()
	if plot_torque:
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(torque_profile)
		plt.show()
	print("walkdone")



#start simulation
with torch.no_grad():
	while True:
		env.set_reference_velocity(kd_tree_env.reference_velocity.astype(np.float32))
		env.reset()
		current_root_pos = np.zeros((1, 3))
		current_root_pos[0, 0] = 0
		current_root_orientation = np.zeros((1, 3, 3))
		current_root_orientation[:, 0, 0] = 1
		current_root_orientation[:, 1, 1] = 1
		current_root_orientation[:, 2, 2] = 1
		obs = env.observe()
		counter = 0

		primitives = [pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
						walk,
						pick_up,
						walk,
						put_down,
		]
		parameters = [
			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "reset": True, "timestep": 0.032},

			{"data": pose_first_walk, "time_limit": 120, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.8, -0.1]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},


			{"data": pose_second_walk, "time_limit": 120, "debug": False, "box_height":0.15, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.1, 0.15]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_third_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[-0.0, 0.0]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_fourth_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.0, 0.0]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "reset": False, "timestep": 0.032},

			{"data": pose_fourth_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.0, 0.1]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_third_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.1, 0.0]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "reset": False, "timestep": 0.032},

			{"data": pose_first_walk, "time_limit": 120, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.4, 0.5]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_fourth_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.2, 0.1]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_third_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.4, -0.8]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_third_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.0, 0.1]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_fourth_walk, "time_limit": 160, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.0, 0.2]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_second_walk, "time_limit": 120, "debug": False, "box_height":0.3, "carry": False, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para": [0.15, 0.0]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},

			{"data": pose_first_walk, "time_limit": 120, "debug": False, "box_height":0.15, "carry": True, "reset": False, "debug_timing": 0, "warming_up_time": 0, "timestep": 0.032, "plot_torque": False, "tuning_para":[0.8, 0.1]},

			{"start_time": 320, "timer":75, "reset": False, "debug": False, "debug_timing": 0, "timestep": 0.032},
		]

		# import ipdb; ipdb.set_trace()
		save_state = np.zeros((100000, 23*7))
		for i in range(len(primitives)):
			print(i)
			primitives[i](**parameters[i])


		import ipdb; ipdb.set_trace()
