import os
import numpy as np
from scipy.spatial.transform import Rotation
# from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree as KDTree 
import sys
sys.path.append('/home/zhaoming/Documents/open_robot/raisim_build/lib')
import raisimpy as raisim

from fairmotion.data import bvh
from fairmotion.core.velocity import MotionWithVelocity

def coordinate_transform(q):
	# q[0:4] = q[[0, 3, 1, 2]]
	# q[1] = -q[1]
	# q[3] = -q[3]
	q[q[:, 0] < 0, :] *= -1
	return q

def process_mocap(mocap_file, phase_file, mirror=False, height_offset=0):
	motion = bvh.load(mocap_file)
	motion = MotionWithVelocity.from_motion(motion)
	motion.compute_velocities()
	phase = np.loadtxt(phase_file)[:, :]
	matrix = motion.to_matrix(local=False)[:, :, :, :]
	local_matrix = motion.to_matrix(local=True)[:, :, : ,:]
	num_frames = matrix.shape[0]
	frame_look_ahead = 30

	if mirror:
		matrix[:, [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22], :, :] = matrix[:, [11,12,13,14, 7,8,9,10, 19,20,21,22, 15,16,17,18], :, :] 
		matrix[:, :, 0, 3] *= -1
		matrix[:, :, 0, 1] *= -1
		matrix[:, :, 0, 2] *= -1
		matrix[:, :, 1, 0] *= -1
		matrix[:, :, 2, 0] *= -1

		local_matrix[:, [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22], :, :] = local_matrix[:, [11,12,13,14, 7,8,9,10, 19,20,21,22, 15,16,17,18], :, :]

		local_matrix[:, :, 0, 3] *= -1
		local_matrix[:, :, 0, 1] *= -1
		local_matrix[:, :, 0, 2] *= -1
		local_matrix[:, :, 1, 0] *= -1
		local_matrix[:, :, 2, 0] *= -1
	pos = matrix[:, :, :3, 3]

	#generate virtual root pose
	root_pos = np.zeros((num_frames, 3))
	root_rot = np.zeros((num_frames, 3, 3))
	root_pos[:, 0] = pos[:, 0, 0]
	root_pos[:, 2] = pos[:, 0, 2]
	root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))  #right hip - left hip, hip_root - chest
	root_rot[:, 1, 0] = 0
	norms = np.apply_along_axis(np.linalg.norm, 1, root_rot[:, :, 0])
	root_rot[:, :, 0] /= norms[:, np.newaxis]
	root_rot[:, :, 1] = np.array([0, 1, 0])
	root_rot[:, :, 2] = np.cross(root_rot[:, :, 0], root_rot[:, :, 1])

	#root velocity
	root_linear_vel = np.zeros((num_frames, 3))
	root_angular_vel = np.zeros((num_frames, 3, 3))
	root_angular_vel[-1, :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	for i in range(num_frames-1):
		root_linear_vel[i] = root_rot[i, :, :].T.dot(root_pos[i+1, :] - root_pos[i, :])
		root_angular_vel[i, :, :] = root_rot[i+1, :, :].dot(root_rot[i, :, :].T)

	#generate bone data w.r.t virtual root
	store_data = np.zeros((num_frames, 23, 3))
	bone_velocity = np.zeros((num_frames, 23, 3))

	#bone position w.r.t the root
	for i in range(23):
		A = np.transpose(root_rot[:, : :], axes=[0, 2, 1])
		x = pos[:, i, :] - root_pos
		# store_data[:, i, :] = np.einsum('ijk, ij->k', A, x)
		store_data[:, i, :] = np.matmul(A, x[:, :, None]).squeeze()
	store_data[:, 0, 1] -= height_offset
	#bone velocity
	for i in range(num_frames-1):
		for j in range(23):
			bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]

	#root bone orientation
	bone_orientation = np.zeros((num_frames, 3, 3))
	for i in range(num_frames-1):
		bone_orientation[i, :, :] = root_rot[i, :, :].T.dot(matrix[i, 0, 0:3, 0:3])

	#kd tree features
	features = 0
	# num_matched_frames = 10
	# frames = []
	# num_phase_matched_frames = 2
	# phase_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	# for i in range(num_matched_frames):
	# 	frames.append(int((1+i) * frame_look_ahead / num_matched_frames - 1))
	# features = np.zeros((num_frames - frame_look_ahead,  6 * num_phase_matched_frames))
	# for i in range(num_frames - frame_look_ahead):
	# 	root_pos_feature = np.zeros(3)

	# 	for j in range(num_phase_matched_frames):
	# 		features[i, 6 * j: 6 * j + 6] = phase[i+phase_frames[j], 0:6].copy()

	#build query data of shape (num_frames-frame_look_ahead) * frame_look_ahead * data_shape
	query_root_linear_vel = np.zeros((num_frames - frame_look_ahead, frame_look_ahead, 3))
	for i in range(num_frames-frame_look_ahead):
		for j in range(frame_look_ahead):
			query_root_linear_vel[i, j, :] = root_linear_vel[i+j, :].copy()

	query_root_angular_vel = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 3, 3))
	for i in range(num_frames-frame_look_ahead):
		for j in range(frame_look_ahead):
			query_root_angular_vel[i, j, :, :] = root_angular_vel[i+j, :, :].copy()


	#root hip location w.r.t virtual root
	query_store_data = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 3))
	for i in range(num_frames-frame_look_ahead):
		for j in range(frame_look_ahead):
			query_store_data[i, j, :] = store_data[i+j, 0, :].copy()

	query_local_matrix = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 23, 3, 3))
	for i in range(num_frames-frame_look_ahead):
		for j in range(frame_look_ahead):
			query_local_matrix[i, j, :, :, :] = local_matrix[i+j, :, 0:3, 0:3].copy()

	#root hip orientation w.r.t the virtual root
	query_bone_orientation = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 3, 3))
	for i in range(num_frames - frame_look_ahead):
		for j in range(frame_look_ahead):
			query_bone_orientation[i, j, :, :] = bone_orientation[i+j, :, :].copy()

	query_phase = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 9))
	# for i in range(num_frames - frame_look_ahead):
	# 	for j in range(frame_look_ahead):
	# 		query_phase[i, j, :] = phase[i+j, :].copy()

	#bone angular velocity
	# query_bone_angular_velocity = 0
	query_bone_angular_velocity = np.zeros((num_frames-frame_look_ahead, frame_look_ahead, 23, 3))
	for i in range(num_frames - frame_look_ahead):
		for j in range(frame_look_ahead):
			for k in range(23):
					query_bone_angular_velocity[i, j, k, :] = motion.vels[i+j].data_local[k][0:3].copy()
	if mirror:
		query_bone_angular_velocity[:, :, [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22], :] = query_bone_angular_velocity[:, :, [11,12,13,14, 7,8,9,10, 19,20,21,22, 15,16,17,18], :]
		query_bone_angular_velocity[:, :, :, 1] *= -1
		query_bone_angular_velocity[:, :, :, 2] *= -1

	print("process done: ", mocap_file)
	
	return query_root_linear_vel, query_root_angular_vel, query_store_data, query_local_matrix, features, query_bone_orientation, query_phase, query_bone_angular_velocity


def set_human_pose(index, hip_pose=None, hip_rot=None):
	reference =  np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
	 0., 0., 0.,])
	base_rot = R.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()
	#base_rot = R.from_quat([0.707, 0.0, 0.0, 0.707]).as_matrix()
		#r = R.from_matrix(matrix[j, 0, 0:3, 0:3])
		#rotation = base_rot.dot(r.as_matrix().T)
		#translation = r.as_matrix().T.dot(matrix[j, 0, 0:3, 3])

	#hip rot
	if hip_pose is None:
		reference[0:3] = base_rot.dot(matrix[j, 0, 0:3, 3])/100#(r.as_matrix().dot(matrix[j, 0, 0:3, 3]) - translation)[[0, 2, 1]] / 100 + np.array([0, 0, 1])
	else:
		reference[0:3] = (hip_pose)[[0, 2, 1]] / 100 - np.array([0, 0, 0.05])
		reference[1] *= -1
	if hip_rot is None:
		hip_rot = R.from_matrix(base_rot.dot(matrix[index, 0, 0:3, 0:3])).as_quat()
	else:
		hip_rot = R.from_matrix(base_rot.dot(hip_rot)).as_quat()
	reference[3:7] = hip_rot[[3, 0, 1, 2]]

	#set left leg
	reference[7:11] = coordinate_transform(R.from_matrix(local_matrix[index, 19, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[11:15] = coordinate_transform(R.from_matrix(local_matrix[index, 20, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[15:19] = coordinate_transform(R.from_matrix(local_matrix[index, 21, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set right leg
	reference[19:23] = coordinate_transform(R.from_matrix(local_matrix[index, 15, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[23:27] = coordinate_transform(R.from_matrix(local_matrix[index, 16, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[27:31] = coordinate_transform(R.from_matrix(local_matrix[index, 17, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set body
	reference[31:35] = coordinate_transform(R.from_matrix(local_matrix[index, 1, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[35:39] = coordinate_transform(R.from_matrix(local_matrix[index, 2, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[39:43] = coordinate_transform(R.from_matrix(local_matrix[index, 3, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[43:47] = coordinate_transform(R.from_matrix(local_matrix[index, 4, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[47:51] = coordinate_transform(R.from_matrix(local_matrix[index, 5, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set left arms
	reference[51:55] = coordinate_transform(R.from_matrix(local_matrix[index, 11, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	reference[55:59] = coordinate_transform(R.from_matrix(local_matrix[index, 12, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[59:63] = coordinate_transform(R.from_matrix(local_matrix[index, 13, 0:3, 0:3]).as_quat()[[3,0,1,2]])

	#set right arms
	reference[63:67] = coordinate_transform(R.from_matrix(local_matrix[index, 7, 0:3, 0:3]).as_quat()[[3,0,2,1]])
	reference[67:71] = coordinate_transform(R.from_matrix(local_matrix[index, 8, 0:3, 0:3]).as_quat()[[3,0,1,2]])
	reference[71:75] = coordinate_transform(R.from_matrix(local_matrix[index, 9, 0:3, 0:3]).as_quat()[[3,0,1,2]])


	virtual_human.setState(reference, np.zeros([virtual_human.getDOF()]))

def build_KD_tree(root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity):
	# root_linear_vel1, root_angular_vel1, store_data1, local_matrix1, features1, bone_orientation1, phase1, bone_velocity1 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Loco/RunTurn1.bvh", "NSM_phase2/RunTurn1.bvh/Phases_Standard.txt")
	# root_linear_vel3, root_angular_vel3, store_data3, local_matrix3, features3, bone_orientation3, phase3, bone_velocity3 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Standard.txt")
	# root_linear_vel4, root_angular_vel4, store_data4, local_matrix4, features4, bone_orientation4, phase4, bone_velocity4 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Standard.txt")
	# root_linear_vel5, root_angular_vel5, store_data5, local_matrix5, features5, bone_orientation5, phase5, bone_velocity5 = process_mocap("Loco/RunSideBack1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel6, root_angular_vel6, store_data6, local_matrix6, features6, bone_orientation6, phase6, bone_velocity6 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel7, root_angular_vel7, store_data7, local_matrix7, features7, bone_orientation7, phase7, bone_velocity7 = process_mocap("Loco/RunSideBack1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Standard.txt")
	# root_linear_vel8, root_angular_vel8, store_data8, local_matrix8, features8, bone_orientation8, phase8, bone_velocity8 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Standard.txt", mirror=False)
	# root_linear_vel9, root_angular_vel9, store_data9, local_matrix9, features9, bone_orientation9, phase9, bone_velocity9 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel10, root_angular_vel10, store_data10, local_matrix10, features10, bone_orientation10, phase10, bone_velocity10 = process_mocap("Loco/RunTurn1.bvh", "NSM_phase2/RunTurn1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel11, root_angular_vel11, store_data11, local_matrix11, features11, bone_orientation11, phase11, bone_velocity11 = process_mocap("Loco/RunSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel12, root_angular_vel12, store_data12, local_matrix12, features12, bone_orientation12, phase12, bone_velocity12 = process_mocap("Loco/RunSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# root_linear_vel13, root_angular_vel13, store_data13, local_matrix13, features13, bone_orientation13, phase13, bone_velocity13 = process_mocap("Carry/CarryFree1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True, height_offset=12)
	# root_linear_vel14, root_angular_vel14, store_data14, local_matrix14, features14, bone_orientation14, phase14, bone_velocity14 = process_mocap("Carry/CarryFree1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=False, height_offset=12)
	
	# root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4, root_linear_vel5, root_linear_vel6, root_linear_vel7, root_linear_vel8, root_linear_vel9, root_linear_vel10, root_linear_vel11, root_linear_vel12, root_linear_vel13, root_linear_vel14))
	# root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4, root_angular_vel5, root_angular_vel6, root_angular_vel7, root_angular_vel8, root_angular_vel9, root_angular_vel10, root_angular_vel11, root_angular_vel12, root_angular_vel13, root_angular_vel14))
	# store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4, store_data5, store_data6, store_data7, store_data8, store_data9, store_data10, store_data11, store_data12, store_data13, store_data14))
	# local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4, local_matrix5, local_matrix6, local_matrix7, local_matrix8, local_matrix9, local_matrix10, local_matrix11, local_matrix12, local_matrix13, local_matrix14))
	# features = 0#np.concatenate((features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12, features13, features14))
	# bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4, bone_orientation5, bone_orientation6, bone_orientation7, bone_orientation8, bone_orientation9, bone_orientation10, bone_orientation11, bone_orientation12, bone_orientation13, bone_orientation14))
	# phase = np.concatenate((phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, phase13, phase14))
	# bone_velocity = 0#np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4, bone_velocity5, bone_velocity6, bone_velocity7, bone_velocity8, bone_velocity9, bone_velocity10, bone_velocity11, bone_velocity12, bone_velocity13, bone_velocity14))
	
	# root_linear_vel1, root_angular_vel1, store_data1, local_matrix1, features1, bone_orientation1, phase1, bone_velocity1 = process_mocap("Carry/CarryFree1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True, height_offset=12)
	# root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Carry/CarryFree1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=False, height_offset=12)
	# root_linear_vel3, root_angular_vel3, store_data3, local_matrix3, features3, bone_orientation3, phase3, bone_velocity3 = process_mocap("Carry/CarryFree2.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=False, height_offset=12)
	# root_linear_vel4, root_angular_vel4, store_data4, local_matrix4, features4, bone_orientation4, phase4, bone_velocity4 = process_mocap("Carry/CarryFree2.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True, height_offset=12)
	# root_linear_vel5, root_angular_vel5, store_data5, local_matrix5, features5, bone_orientation5, phase5, bone_velocity5 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel6, root_angular_vel6, store_data6, local_matrix6, features6, bone_orientation6, phase6, bone_velocity6 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False)
	# root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4, root_linear_vel5, root_linear_vel6))
	# root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4, root_angular_vel5, root_angular_vel6))
	# store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4, store_data5, store_data6))
	# local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4, local_matrix5, local_matrix6))
	# features = np.concatenate((features1, features2, features3, features4, features5, features6))
	# bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4, bone_orientation5, bone_orientation6))
	# phase = np.concatenate((phase1, phase2, phase3, phase4, phase5, phase6))
	# bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4, bone_velocity5, bone_velocity6))
	
	# root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Loco/RunSideBack1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Standard.txt")
	# root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2))
	# root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2))
	# store_data = np.concatenate((store_data1, store_data2))
	# local_matrix = np.concatenate((local_matrix1, local_matrix2))
	# features = np.concatenate((features1, features2,))
	# bone_orientation = np.concatenate((bone_orientation1, bone_orientation2))
	# phase = np.concatenate((phase1, phase2))
	# bone_velocity = np.concatenate((bone_velocity1, bone_velocity2))
	# print(root_linear_vel1.shape, root_linear_vel2.shape)

	# feature_mean = np.mean(features, axis=0)
	# feature_std = np.std(features, axis=0)
	# feature_mean[:] = 0
	# feature_std[:] = 1
	# features = (features[:-1, :] - feature_mean[:]) / feature_std[:]
	feature_mean = np.zeros(10)
	feature_std = np.ones(10)
	features = np.ones((100, 10))
	tree = KDTree(features)

	# local_matrix_mean = np.mean(local_matrix, axis=0)
	# local_matrix_std = np.std(local_matrix, axis=0)

	return tree, feature_mean, feature_std

def tree_query(tree, current_phase, num_query, feature_mean, feature_std, noise=np.zeros((400, 6))):
	num_matched_frames = 10
	num_phase_matched_frames = 2
	query_feature = np.zeros((num_query, num_phase_matched_frames  * 6))
	phase_frames = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10]

	phase_amp = np.zeros((num_query, 3))
	phase_offset = np.zeros((num_query, 3))
	phase_speed = np.zeros((num_query, 3))
	for i in range(3):
		phase_amp[:, i] = np.sqrt(current_phase[:, i * 2]**2 + current_phase[:, i * 2 + 1]**2)
		phase_offset[:, i] = np.arctan2(current_phase[:, i*2+1]/phase_amp[:, i], current_phase[:, i*2]/phase_amp[:, i])
		phase_speed[:, i] = current_phase[:, i+6] / 60 * 2 * np.pi

	# print("before", phase_amp[0])
	phase_offset += noise[:, -9:-6] * 0.5
	phase_speed += noise[:, -6:-3] * 1.0  #1.5 for target
	# phase_speed *= (1 + noise[:, -6:-3] * 0.5)
	phase_speed = np.clip(phase_speed, 0.001, 100)
	phase_amp *= (1 + noise[:, -3:] * 1.5)  #0.2 for speed 1,2 0.8 for speed 3
	# phase_amp += (noise[:, -3:] * 1)
	phase_amp = np.clip(phase_amp, 0.0, 100)

	# print("after", phase_amp[0])

	# phase_speed[:] = 0
	# print(phase_speed[0, :], noise[0, -6:-3])
	# phase_amp[:] = 0

	for i in range(num_phase_matched_frames):
		for j in range(3):
			# if i == 0:
			# 	query_feature[:, i * 6 + j * 2] = current_phase[:, j * 2].copy()
			# 	query_feature[:, i * 6 + j * 2 + 1] = current_phase[:, j * 2 + 1].copy()
			# else:
			query_feature[:, i * 6 + j * 2] = phase_amp[:, j] * np.cos(phase_offset[:, j] - phase_speed[:, j] * phase_frames[i])
			query_feature[:, i * 6 + j * 2 + 1] = phase_amp[:, j] * np.sin(phase_offset[:, j] - phase_speed[:, j] * phase_frames[i])

	query_feature = (query_feature-feature_mean) / feature_std
	dist, ind = tree.query(query_feature, k=1, workers=-1)
	return dist, ind


class KD_tree_Env:
	def __init__(self, num_envs):
		print("KD tree Env created.")
		self.num_envs = num_envs
		self.starting_frame = 320
		self.pick_and_place = False

		#build initial dataset
		self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.bone_velocity = process_mocap("Loco/CircularWalking1.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False)
		root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Loco/CircularWalking1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)

		#use for carry
		# self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.bone_velocity = process_mocap("Carry/CarryFree6.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False, height_offset=12)
		# root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Carry/CarryFree6.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)

		#use for walking
		# self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.bone_velocity = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False)
		# root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
		
		self.root_linear_vel = np.concatenate((self.root_linear_vel, root_linear_vel))
		self.root_angular_vel = np.concatenate((self.root_angular_vel, root_angular_vel))
		self.store_data = np.concatenate((self.store_data, store_data))
		self.local_matrix = np.concatenate((self.local_matrix, local_matrix))
		self.features = 0#np.concatenate((self.features, features,))
		self.bone_orientation = np.concatenate((self.bone_orientation, bone_orientation))
		self.phase = np.concatenate((self.phase, phase))
		self.bone_velocity = np.concatenate((self.bone_velocity, bone_velocity))

		# self.dataset_name = ["Loco/CircularWalking1.bvh", "Loco/RegularWalking1.bvh", "Loco/RunRandom.bvh", "Loco/RunSideBack1.bvh", "Loco/RunSideBack2.bvh", "Loco/RunSideBack3.bvh", "Loco/RunTurn1.bvh", "Loco/RunTurn2.bvh", "Loco/RunTurn3.bvh", "Loco/Sidestep1.bvh", "Loco/SuddenWalking1.bvh", "Loco/WalkRandom.bvh", "Loco/WalkSideBack1.bvh", "Loco/WalkSideBack2.bvh", "Loco/WalkSideBack3.bvh", "Loco/WalkTurn1.bvh", "Loco/WalkTurn2.bvh", "Loco/WalkTurn3.bvh", "Crouch/Crouch1.bvh", "Crouch/Crouch2.bvh", "Crouch/Crouch3.bvh", "Jump/Jump1.bvh", "Jump/Jump2.bvh", "Jump/Jump3.bvh"]
		self.frame_look_ahead = 30
		self.dataset_size = [3751-self.frame_look_ahead, 3751-self.frame_look_ahead]
		self.dataset_name = ["Loco/CircularWalking1.bvh"]
		
		# dataset for Carry
		# self.dataset_name = ["Loco/WalkRandom.bvh"]
		# self.frame_look_ahead = 30
		# self.dataset_size = [3751-self.frame_look_ahead, 3751-self.frame_look_ahead]

		# dataset for Pick and Place
		# self.dataset_name = ["Carry/CarryFree2.bvh"]
		# self.frame_look_ahead = 30
		# self.dataset_size = [1987-self.frame_look_ahead, 1987-self.frame_look_ahead]


		for i in range(1, len(self.dataset_name)):
			self.addDataset(self.dataset_name[i], "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False)
			self.addDataset(self.dataset_name[i], "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)

		self.build_dataset_for_RL()

		self.sim_index_to_bvh_index = [19, 20, 21, 15, 16, 17, 1, 2, 3, 4, 5, 11, 12, 13, 7, 8, 9]

		if self.pick_and_place:
			self.root_linear_vel[460:560, :, :] = self.root_linear_vel[460, :, :]
			self.root_angular_vel[460:560, :, :, :] = self.root_angular_vel[460, :, :, :]
			self.store_data[460:560, :] = self.store_data[460, :]
			self.local_matrix[460:560, :] = self.local_matrix[460, :]
			self.bone_orientation[460:560, :] = self.bone_orientation[460, :]

			self.root_linear_vel[360:410, :, :] = self.root_linear_vel[410, :, :]
			self.root_angular_vel[360:410, :, :, :] = self.root_angular_vel[410, :, :, :]
			self.store_data[360:410, :] = self.store_data[410, :]
			self.local_matrix[360:410, :] = self.local_matrix[410, :]
			self.bone_orientation[360:410, :] = self.bone_orientation[410, :]

			self.root_linear_vel[0:560, 0:5, :] = 0
			self.root_angular_vel[0:560, 0:5, :, :] = 0
			self.root_angular_vel[0:560, 0:5, 0, 0] = 1
			self.root_angular_vel[0:560, 0:5, 1, 1] = 1
			self.root_angular_vel[0:560, 0:5, 2, 2] = 1

	def addDataset(self, datafile, phasefile, mirror=False):
		root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap(datafile, phasefile, mirror=mirror)
		self.dataset_size.append(root_linear_vel.shape[0])
		self.root_linear_vel = np.concatenate((self.root_linear_vel, root_linear_vel))
		self.root_angular_vel = np.concatenate((self.root_angular_vel, root_angular_vel))
		self.store_data = np.concatenate((self.store_data, store_data))
		self.local_matrix = np.concatenate((self.local_matrix, local_matrix))
		self.features = 0#np.concatenate((self.features, features,))
		self.bone_orientation = np.concatenate((self.bone_orientation, bone_orientation))
		self.phase = np.concatenate((self.phase, phase))
		self.bone_velocity = 0#np.concatenate((self.bone_velocity, bone_velocity))
	
	def build_dataset_for_RL(self):
		self.tree, self.feature_mean, self.feature_std = build_KD_tree(self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.bone_velocity)

		self.current_phase = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (self.num_envs, 1))
		self.next_phase = np.tile(self.phase[self.starting_frame, 5:20:5, 0:9].copy().flatten(), (self.num_envs, 1))
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

	def reset(self):
		self.current_phase = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (self.num_envs, 1))
		self.next_phase = np.tile(self.phase[self.starting_frame, 5, 0:9].copy(), (self.num_envs, 1))
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

		self.reference[:, 0:3] = np.matmul(self.base_rot, (self.current_root_pos + self.store_data[np.ones(self.num_envs, dtype=np.int) * self.starting_frame, 0, :])[:, :, None]).squeeze() / 100
		self.reference[:, 3:7] = Rotation.from_matrix(np.matmul(self.base_rot, np.matmul(self.current_root_orientation, self.bone_orientation[np.ones(self.num_envs, dtype=np.int) * self.starting_frame, 0, :]))).as_quat()[:,[3,0,1,2]]
		self.set_reference(np.ones(self.num_envs, dtype=np.int)*self.starting_frame)
		self.set_reference_velocity(np.ones(self.num_envs, dtype=np.int)*self.starting_frame)

	def set_reference(self, index, t=0):
		self.reference[:, 7:11] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 19, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 11:15] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t, 20, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 15:19] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  21, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 19:23] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  15, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 23:27] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  16, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 27:31] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  17, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 31:35] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  1, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 35:39] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  2, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 39:43] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  3, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 43:47] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  4, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 47:51] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  5, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 51:55] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  11, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		self.reference[:, 55:59] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  12, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 59:63] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  13, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 63:67] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  7, 0:3, 0:3]).as_quat()[:, [3,0,2,1]])
		self.reference[:, 67:71] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  8, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		self.reference[:, 71:75] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, t,  9, 0:3, 0:3]).as_quat()[:, [3,0,1,2]])

		self.reference[:, 63:75] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, 0.13, 0.438, 0.85, -0.2, 0.35, 0.34])
		self.reference[:, 51:63] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, -0.13, -0.438, 0.85, -0.2, -0.35, -0.34])

	def set_reference_velocity(self, index, t=0):
		#set left leg
		self.reference_velocity[:, 6:9] = self.bone_velocity[index, t, 19, :]
		self.reference_velocity[:, 9:12] = self.bone_velocity[index, t, 20, :]
		self.reference_velocity[:, 12:15] = self.bone_velocity[index, t, 21, :]
		self.reference_velocity[:, 15:18] = self.bone_velocity[index, t, 15, :]
		self.reference_velocity[:, 18:21] = self.bone_velocity[index, t, 16, :]
		self.reference_velocity[:, 21:24] = self.bone_velocity[index, t, 17, :]
		self.reference_velocity[:, 24:27] = self.bone_velocity[index, t, 1, :]
		self.reference_velocity[:, 27:30] =self.bone_velocity[index, t, 2, :]
		self.reference_velocity[:, 30:33] = self.bone_velocity[index, t, 3, :]
		self.reference_velocity[:, 33:36] = self.bone_velocity[index, t, 4, :]
		self.reference_velocity[:, 36:39] = self.bone_velocity[index, t, 5, :]
		self.reference_velocity[:, 39:42] = self.bone_velocity[index, t, 11, :]
		self.reference_velocity[:, 42:45] = self.bone_velocity[index, t, 12, :]
		self.reference_velocity[:, 45:48] = self.bone_velocity[index, t, 13, :]
		self.reference_velocity[:, 48:51] = self.bone_velocity[index, t, 7, :]
		self.reference_velocity[:, 51:54] = self.bone_velocity[index, t, 8, :]
		self.reference_velocity[:, 54:57] = self.bone_velocity[index, t, 9, :]
		self.reference_velocity[:, 0:3] = self.root_linear_vel[index, t, :, None].squeeze() / 100.0 * 60.0

	def sample_joint_sequence(self, index, sequence_length=10):
		output = np.zeros((self.num_envs, sequence_length, 17, 4))
		for i in range(17):
			for j in range(sequence_length):
				output[:, j, i, :] = coordinate_transform(Rotation.from_matrix(self.local_matrix[index, j,  self.sim_index_to_bvh_index[i], 0:3, 0:3]).as_quat()[:, [3,0,1,2]])
		return output.copy()

	def sample_root_linear_sequence(self, index, sequence_length=10):
		output = np.zeros((self.num_envs, sequence_length, 3))
		for i in range(sequence_length):
			output[:, i] = self.root_linear_vel[index, i, :]
		return output.copy()

	def sample_root_angular_sequence(self, index, sequence_length=10):
		output = np.zeros((self.num_envs, sequence_length, 3, 3))
		for i in range(sequence_length):
			output[:, i, :, :] = self.root_angular_vel[index, i, :, :]
		return output.copy()

	def sample_bone_orientation_sequence(self, index, sequence_length=10):
		output = np.zeros((self.num_envs, sequence_length, 3, 3))
		for i in range(sequence_length):
			output[:, i, :, :] = self.bone_orientation[index, i, :, :]
		return output.copy()

	def sample_store_data_sequence(self, index, sequence_length=10):
		output = np.zeros((self.num_envs, sequence_length, 3))
		for i in range(sequence_length):
			output[:, i] = self.store_data[index, i, :]
		return output.copy()