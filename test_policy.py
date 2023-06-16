import json
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import box_manipulation_release
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher

from process_mocap import KD_tree_Env, tree_query, coordinate_transform
from scipy.spatial.transform import Rotation

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
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def process_pose_output(pose):
   length = pose.shape[1]
   joint_output = torch.zeros((length, 4 * 17))
   for i in range(length):
      for j in range(17):
         d6_rotation = torch.zeros((1, 6))
         d6_rotation[0, 0:3] = pose[0, i, j * 6: j * 6 + 3]
         d6_rotation[0, 3:6] = pose[0, i, j * 6 + 3: j * 6 + 6]
         rotation_matrix = rotation_6d_to_matrix(d6_rotation)
         quaternion = matrix_to_quaternion(rotation_matrix)
         joint_output[i, j * 4: j * 4 + 4] = quaternion.clone()

   root_linear_output = torch.zeros((length, 3))
   for i in range(length):
      root_linear_output[i, :] = pose[0, i, 17 * 6: 17 * 6+3]

   root_angular_output = torch.zeros((length, 3, 3))
   for i in range(length):
      root_angular_output[i, : ,:] = rotation_6d_to_matrix(pose[0, i, 17 * 6 + 3: 17*6+3+6])

   bone_orientation_output = torch.zeros((length, 3, 3))
   for i in range(length):
      bone_orientation_output[i, :, :] = rotation_6d_to_matrix(pose[0, i, 17 * 6 + 3 + 6: 17*6+3+12])
   
   return joint_output, root_linear_output, root_angular_output, bone_orientation_output

def simulate_from_pose_data():
   import ipdb; ipdb.set_trace()
   pose_data = torch.load("pose_tensor_walk_forward.pt")
   joint, linear_vel, angular_vel, bone_orientation = process_pose_output(pose_data)
   current_phase = np.zeros((1, 9))
   next_phase = np.zeros((1, 27))
   base_rot_quat = np.zeros((1, 4))
   base_rot_quat[:, 0] = 0.707
   base_rot_quat[:, 3] = 0.707
   base_rot = Rotation.from_quat(base_rot_quat).as_matrix()

   root_offset = np.zeros((1, 3))
   root_offset[0, 1] = 1.
   # import ipdb; ipdb.set_trace()

   while True:
      current_root_pos = np.zeros((1, 3))
      current_root_orientation = np.zeros((1, 3, 3))
      current_root_orientation[:, 0, 0] = 1
      current_root_orientation[:, 1, 1] = 1
      current_root_orientation[:, 2, 2] = 1
      env.reset()
      obs = env.observe()
      for t in range(240):
         reference = np.zeros((1, 75))
         reference[0, 2] = 1.35
         reference[0, 3:7] = 0.5
         for i in range(17):
            reference[0, 7 + 4 * i: 11 + 4 * i] = joint[t][4*i:4*i+4].cpu().numpy()
         
         current_root_pos[0, 0:3]  += np.matmul(current_root_orientation[0, :, :], linear_vel[t, :, None].cpu().numpy()).squeeze()
         current_root_pos[0, 1] = 0
         current_root_orientation[0, :, :] = np.matmul(angular_vel[t, :, :], current_root_orientation[0, :, :])

         reference[0, 0:3] = np.matmul(base_rot, (current_root_pos + root_offset)[:, :, None]).squeeze()
         reference[0, 2] -= 0.05
         reference[0, 3:7] = Rotation.from_matrix(np.matmul(base_rot, np.matmul(current_root_orientation, bone_orientation[t:t+1, :, :]))).as_quat()[:,[3,0,1,2]]
         
         env.set_reference(np.concatenate((current_phase, next_phase, reference), axis=1).astype(np.float32))
         with torch.no_grad():
            dynamic_act = dynamic_model.sample_best_actions(obs)
         obs, rew, done, _ = env.step(dynamic_act * 0)
         import ipdb; ipdb.set_trace()
         print(dynamic_act)
         import time; time.sleep(0.01)

def simulate_from_KD_tree():
   # env.setTask()
   print(kd_tree_env.reference)
   env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
   env.reset()
   obs = env.observe()
   # kin_state = env.get_reference()
   print(obs[0, :])
   average_gating = np.zeros(8)
   average_gating_sum = 0
   phase = np.ones(2, dtype=np.int)
   step = 0
   save_phase = np.zeros((200, 6))
   use_kinematics = False
   ind = np.zeros(1, dtype=np.int) + kd_tree_env.starting_frame
   save_counter = 0
   save_state = np.zeros((3000, 24*7))
   torque_profile = np.zeros(3000)
   previous_act = torch.zeros(1, 53, device=device)
   for i in range(10000):
      dynamic_done = np.zeros(1, dtype=np.int)
      for t in range(2):
         kd_tree_env.current_phase = kd_tree_env.phase[ind, t].copy()
         kd_tree_env.next_phase = kd_tree_env.phase[ind, 5:20:5].copy().reshape(1, -1)
         save_phase[(i * 5 + t)%200, :] = kd_tree_env.current_phase[0, 0:6]
         kd_tree_env.set_reference(ind, t)
         kd_tree_env.current_root_pos += np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.root_linear_vel[ind, t, :, None]).squeeze()
         kd_tree_env.current_root_orientation[:, :, :] = np.matmul(kd_tree_env.root_angular_vel[ind, t, :, :], kd_tree_env.current_root_orientation)
         kd_tree_env.reference[:, 0:3] = np.matmul(kd_tree_env.base_rot, (kd_tree_env.current_root_pos + kd_tree_env.store_data[ind, t, :])[:, :, None]).squeeze() / 100
         kd_tree_env.reference[:, 3:7] = Rotation.from_matrix(np.matmul(kd_tree_env.base_rot, np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.bone_orientation[ind, t, :]))).as_quat()[:,[3,0,1,2]]
         kd_tree_env.reference[kd_tree_env.reference[:, 3] < 0, 3:7] *= -1


      if not use_kinematics:
         # obs[0, 0] = i * 0.016
         env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
         with torch.no_grad():
            dynamic_act = dynamic_model.sample_best_actions(obs)
         obs, rew, done, _ = env.step(dynamic_act)
         dynamic_done = dynamic_done + done.cpu().numpy()
         import time; time.sleep(0.03)
         save_state[save_counter, :] = env.get_state()[0, :].cpu().numpy()
         torque_profile[save_counter] = (0.5*dynamic_act[0, 6].cpu().numpy()+0.5*previous_act[0, 6].cpu().numpy())
         previous_act = 0.5 * dynamic_act + previous_act * 0.5
         save_counter += 1
         print(dynamic_act[:, 6:9])
         print(rew[:])
         #print(ind[0], rew[0])
      ind += 2

      if dynamic_done[0] > 0 or save_counter == 300:
         import ipdb; ipdb.set_trace()
         kd_tree_env.reset()
         ind = np.zeros(1, dtype=np.int) + kd_tree_env.starting_frame
         env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.reference), axis=1).astype(np.float32))
         env.set_reference_velocity(kd_tree_env.reference_velocity.astype(np.float32))
         env.reset()
         obs = env.observe()
         import matplotlib.pyplot as plt
         plt.figure()
         plt.plot(torque_profile[60:save_counter])
         save_counter = 0
         plt.show()


if __name__ == '__main__':
   seed = 3#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # config
   cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

   # create environment from the configuration file
   env = VecEnv(box_manipulation_release.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
   kd_tree_env = KD_tree_Env(1)
   print("env_created")

   dynamic_model = ActorCriticNetMann(env.observation_space.shape[0], env.action_space.shape[0], [128, 128])
   dynamic_model.load_state_dict(torch.load("walk_model.pt")) 
   dynamic_model.cuda()

   simulate_from_KD_tree()