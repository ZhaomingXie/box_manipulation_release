if __name__ == '__main__':
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
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
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
   num_envs = 1
   kd_tree_env = KD_tree_Env(num_envs)
   print("env_created")

   dynamic_model = ActorCriticNetMann(env.observation_space.shape[0], env.action_space.shape[0], [128, 128])
   dynamic_model.load_state_dict(torch.load("pick_and_place_model.pt"))
   dynamic_model.cuda()

   # env.setTask()
   env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
   env.reset()
   obs = env.observe()
   kin_state = env.get_reference()
   print(obs[0, :])
   average_gating = np.zeros(8)
   average_gating_sum = 0
   phase = np.ones(2, dtype=np.int)
   step = 0
   save_phase = np.zeros((200, 6))
   use_kinematics = False
   ind = np.zeros(num_envs, dtype=np.int) + kd_tree_env.starting_frame
   i = 0
   state_counter = 0
   save_state = np.zeros((10000, 23*7))
   torque_profile = np.zeros(10000)
   while True:
      counter = 0
      dynamic_done = np.zeros(num_envs, dtype=np.int)
      for t in range(5):
         kd_tree_env.current_phase = kd_tree_env.phase[ind, t].copy()
         kd_tree_env.next_phase = kd_tree_env.phase[ind, 5:20:5].copy().reshape(num_envs, -1)
         save_phase[(i * 5 + t)%200, :] = kd_tree_env.current_phase[0, 0:6]
         kd_tree_env.set_reference(ind, t)
         kd_tree_env.current_root_pos += np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.root_linear_vel[ind, t, :, None]).squeeze()
         kd_tree_env.current_root_orientation[:, :, :] = np.matmul(kd_tree_env.root_angular_vel[ind, t, :, :], kd_tree_env.current_root_orientation)
         # print("root orientation ", kd_tree_env.current_root_orientation[0, : ,:])
         kd_tree_env.reference[:, 0:3] = np.matmul(kd_tree_env.base_rot, (kd_tree_env.current_root_pos + kd_tree_env.store_data[ind, t, :])[:, :, None]).squeeze() / 100
         kd_tree_env.reference[:, 3:7] = Rotation.from_matrix(np.matmul(kd_tree_env.base_rot, np.matmul(kd_tree_env.current_root_orientation, kd_tree_env.bone_orientation[ind, t, :]))).as_quat()[:,[3,0,1,2]]
         kd_tree_env.reference[kd_tree_env.reference[:, 3] < 0, 3:7] *= -1

         if not use_kinematics:
            env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
            with torch.no_grad():
               dynamic_act = dynamic_model.sample_best_actions(obs)
            obs, rew, done, _ = env.step(dynamic_act)
            # if ind[0] == 1001:
            # import ipdb; ipdb.set_trace()
            dynamic_done = dynamic_done + done.cpu().numpy()
            torque_profile[state_counter] = dynamic_act[0, 6].cpu().numpy()
            import time; time.sleep(0.01)
            print(ind, t)
            save_state[state_counter, :] = env.get_state()[0, :].cpu().numpy()
            state_counter += 1

      ind += 5
      if ind[0] >= 560 and state_counter <= 720:
         ind[:] = 320
         time_limit_done = env.reset_time_limit()
         if time_limit_done[0] > 0:
            kd_tree_env.reset()
            ind = np.zeros(num_envs, dtype=np.int) + kd_tree_env.starting_frame
            env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
            env.set_reference_velocity(kd_tree_env.reference_velocity.astype(np.float32))

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(torque_profile[60:state_counter])
            plt.show()
            import ipdb; ipdb.set_trace()
            env.reset()
            state_counter = 0
            obs = env.observe()
      # print(obs)
      kin_state = env.get_reference()
      if dynamic_done[0] > 0 or state_counter >= 720:
         import ipdb; ipdb.set_trace()
         kd_tree_env.reset()
         ind = np.zeros(num_envs, dtype=np.int) + kd_tree_env.starting_frame
         env.set_reference(np.concatenate((kd_tree_env.current_phase, kd_tree_env.next_phase, kd_tree_env.reference), axis=1).astype(np.float32))
         env.set_reference_velocity(kd_tree_env.reference_velocity.astype(np.float32))
         env.reset()
         obs = env.observe()

      import time; time.sleep(0.032)

      print(ind[0], rew[0], done[0])