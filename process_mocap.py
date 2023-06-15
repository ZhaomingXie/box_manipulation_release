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

def process_mocap(mocap_file, phase_file, mirror=False):
	motion = bvh.load(mocap_file)
	motion = MotionWithVelocity.from_motion(motion)
	motion.compute_velocities()
	phase = np.loadtxt(phase_file)[:, :]
	matrix = motion.to_matrix(local=False)[:, :, :, :]
	local_matrix = motion.to_matrix(local=True)[:, :, : ,:]
	num_frames = matrix.shape[0]
	frame_look_ahead = 3

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
	root_rot[:, :, 0] = np.cross((pos[:, 15, :] - pos[:, 19, :]), (pos[:, 0, :] - pos[:, 1, :]))
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
	#bone velocity
	for i in range(num_frames-1):
		for j in range(23):
			bone_velocity[i, j, :] = store_data[i+1, j, :] - store_data[i, j, :]

	#root bone orientation
	bone_orientation = np.zeros((num_frames, 3, 3))
	for i in range(num_frames-1):
		bone_orientation[i, :, :] = root_rot[i, :, :].T.dot(matrix[i, 0, 0:3, 0:3])

	#kd tree features
	num_matched_frames = 10
	frames = []
	num_phase_matched_frames = 2
	phase_frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	for i in range(num_matched_frames):
		frames.append(int((1+i) * 60 / num_matched_frames - 1))
	features = np.zeros((num_frames - 60,  6 * num_phase_matched_frames))
	# for i in range(num_frames - 60):
	# 	root_pos_feature = np.zeros(3)

	# 	for j in range(num_phase_matched_frames):
	# 		features[i, 6 * j: 6 * j + 6] = phase[i+phase_frames[j], 0:6].copy()

	#build query data of shape (num_frames-60) * 60 * data_shape
	query_root_linear_vel = np.zeros((num_frames - 60, frame_look_ahead, 3))
	for i in range(num_frames-60):
		for j in range(frame_look_ahead):
			query_root_linear_vel[i, j, :] = root_linear_vel[i+j, :].copy()

	query_root_angular_vel = np.zeros((num_frames-60, frame_look_ahead, 3, 3))
	for i in range(num_frames-60):
		for j in range(frame_look_ahead):
			query_root_angular_vel[i, j, :, :] = root_angular_vel[i+j, :, :].copy()


	#root hip location w.r.t virtual root
	query_store_data = np.zeros((num_frames-60, frame_look_ahead, 3))
	for i in range(num_frames-60):
		for j in range(frame_look_ahead):
			query_store_data[i, j, :] = store_data[i+j, 0, :].copy()

	query_local_matrix = np.zeros((num_frames-60, frame_look_ahead, 23, 4, 4))
	for i in range(num_frames-60):
		for j in range(frame_look_ahead):
			query_local_matrix[i, j, :, :, :] = local_matrix[i+j, :, :, :].copy()

	#root hip orientation w.r.t the virtual root
	query_bone_orientation = np.zeros((num_frames-60, frame_look_ahead, 3, 3))
	for i in range(num_frames - 60):
		for j in range(frame_look_ahead):
			query_bone_orientation[i, j, :, :] = bone_orientation[i+j, :, :].copy()

	query_phase = np.zeros((num_frames-60, frame_look_ahead, 9))
	# for i in range(num_frames - 60):
	# 	for j in range(frame_look_ahead):
	# 		query_phase[i, j, :] = phase[i+j, :].copy()

	#bone angular velocity
	query_bone_angular_velocity = np.zeros((num_frames-60, frame_look_ahead, 23, 3))
	for i in range(num_frames - 60):
		for j in range(frame_look_ahead):
			for k in range(23):
					query_bone_angular_velocity[i, j, k, :] = motion.vels[i+j].data_local[k][0:3].copy()
	if mirror:
		query_bone_angular_velocity[:, :, [7,8,9,10, 11,12,13,14, 15,16,17,18, 19,20,21,22], :] = query_bone_angular_velocity[:, :, [11,12,13,14, 7,8,9,10, 19,20,21,22, 15,16,17,18], :]
		query_bone_angular_velocity[:, :, :, 1] *= -1
		query_bone_angular_velocity[:, :, :, 2] *= -1
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

def build_KD_tree():
	# root_linear_vel1, root_angular_vel1, store_data1, local_matrix1, features1, bone_orientation1, phase1, bone_velocity1 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Loco/RunTurn1.bvh", "NSM_phase2/RunTurn1.bvh/Phases_Standard.txt")
	# root_linear_vel3, root_angular_vel3, store_data3, local_matrix3, features3, bone_orientation3, phase3, bone_velocity3 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Standard.txt")
	# root_linear_vel4, root_angular_vel4, store_data4, local_matrix4, features4, bone_orientation4, phase4, bone_velocity4 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Standard.txt")
	# root_linear_vel5, root_angular_vel5, store_data5, local_matrix5, features5, bone_orientation5, phase5, bone_velocity5 = process_mocap("Loco/RunSideBack1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel6, root_angular_vel6, store_data6, local_matrix6, features6, bone_orientation6, phase6, bone_velocity6 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel7, root_angular_vel7, store_data7, local_matrix7, features7, bone_orientation7, phase7, bone_velocity7 = process_mocap("Loco/RunSideBack1.bvh", "NSM_phase2/RunSideBack1.bvh/Phases_Standard.txt")
	# root_linear_vel8, root_angular_vel8, store_data8, local_matrix8, features8, bone_orientation8, phase8, bone_velocity8 = process_mocap("Loco/RunRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel9, root_angular_vel9, store_data9, local_matrix9, features9, bone_orientation9, phase9, bone_velocity9 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel10, root_angular_vel10, store_data10, local_matrix10, features10, bone_orientation10, phase10, bone_velocity10 = process_mocap("Loco/RunTurn1.bvh", "NSM_phase2/RunTurn1.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel11, root_angular_vel11, store_data11, local_matrix11, features11, bone_orientation11, phase11, bone_velocity11 = process_mocap("Loco/RunSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel12, root_angular_vel12, store_data12, local_matrix12, features12, bone_orientation12, phase12, bone_velocity12 = process_mocap("Loco/RunSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# root_linear_vel13, root_angular_vel13, store_data13, local_matrix13, features13, bone_orientation13, phase13, bone_velocity13 = process_mocap("Jump/Jump2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# root_linear_vel14, root_angular_vel14, store_data14, local_matrix14, features14, bone_orientation14, phase14, bone_velocity14 = process_mocap("Jump/Jump2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	# root_linear_vel15, root_angular_vel15, store_data15, local_matrix15, features15, bone_orientation15, phase15, bone_velocity15 = process_mocap("Crouch/Crouch2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# root_linear_vel16, root_angular_vel16, store_data16, local_matrix16, features16, bone_orientation16, phase16, bone_velocity16 = process_mocap("Crouch/Crouch2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	# root_linear_vel17, root_angular_vel17, store_data17, local_matrix17, features17, bone_orientation17, phase17, bone_velocity17 = process_mocap("Door/PushDoor8.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	# root_linear_vel18, root_angular_vel18, store_data18, local_matrix18, features18, bone_orientation18, phase18, bone_velocity18 = process_mocap("Door/PushDoor8.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# # print(root_linear_vel1.shape, root_linear_vel2.shape, root_linear_vel3.shape, root_linear_vel4.shape, root_linear_vel5.shape, root_linear_vel6.shape)
	# # print(root_linear_vel7.shape, root_linear_vel8.shape, root_linear_vel9.shape, root_linear_vel10.shape)
	# print(root_linear_vel12.shape)
	
	# root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4, root_linear_vel5, root_linear_vel6, root_linear_vel7, root_linear_vel8, root_linear_vel9, root_linear_vel10, root_linear_vel11, root_linear_vel12, root_linear_vel13, root_linear_vel14, root_linear_vel15, root_linear_vel16, root_linear_vel17, root_linear_vel18))
	# root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4, root_angular_vel5, root_angular_vel6, root_angular_vel7, root_angular_vel8, root_angular_vel9, root_angular_vel10, root_angular_vel11, root_angular_vel12, root_angular_vel13, root_angular_vel14, root_angular_vel15, root_angular_vel16, root_angular_vel17, root_angular_vel18))
	# store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4, store_data5, store_data6, store_data7, store_data8, store_data9, store_data10, store_data11, store_data12, store_data13, store_data14, store_data15, store_data16,store_data17, store_data18))
	# local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4, local_matrix5, local_matrix6, local_matrix7, local_matrix8, local_matrix9, local_matrix10, local_matrix11, local_matrix12, local_matrix13, local_matrix14, local_matrix15, local_matrix16, local_matrix17, local_matrix18))
	# features = np.concatenate((features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12, features13, features14, features15, features16, features17, features18))
	# bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4, bone_orientation5, bone_orientation6, bone_orientation7, bone_orientation8, bone_orientation9, bone_orientation10, bone_orientation11, bone_orientation12, bone_orientation13, bone_orientation14, bone_orientation15, bone_orientation16, bone_orientation17, bone_orientation18))
	# phase = np.concatenate((phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, phase13, phase14, phase15, phase16, phase17, phase18))
	# bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4, bone_velocity5, bone_velocity6, bone_velocity7, bone_velocity8, bone_velocity9, bone_velocity10, bone_velocity11, bone_velocity12, bone_velocity13, bone_velocity14, bone_velocity15, bone_velocity16, bone_velocity17, bone_velocity18))
	

	# root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Standard.txt", mirror=False)
	# print(root_linear_vel.shape)

	root_linear_vel1, root_angular_vel1, store_data1, local_matrix1, features1, bone_orientation1, phase1, bone_velocity1 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Standard.txt")
	root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Standard.txt")
	root_linear_vel3, root_angular_vel3, store_data3, local_matrix3, features3, bone_orientation3, phase3, bone_velocity3 = process_mocap("Loco/WalkSideBack1.bvh", "NSM_phase2/WalkSideBack1.bvh/Phases_Mirrored.txt", mirror=True)
	root_linear_vel4, root_angular_vel4, store_data4, local_matrix4, features4, bone_orientation4, phase4, bone_velocity4 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Mirrored.txt", mirror=True)
	root_linear_vel5, root_angular_vel5, store_data5, local_matrix5, features5, bone_orientation5, phase5, bone_velocity5 = process_mocap("Loco/WalkTurn1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Mirrored.txt", mirror=True)
	root_linear_vel6, root_angular_vel6, store_data6, local_matrix6, features6, bone_orientation6, phase6, bone_velocity6 = process_mocap("Loco/WalkTurn1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel7, root_angular_vel7, store_data7, local_matrix7, features7, bone_orientation7, phase7, bone_velocity7 = process_mocap("Loco/CircularWalking1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel8, root_angular_vel8, store_data8, local_matrix8, features8, bone_orientation8, phase8, bone_velocity8 = process_mocap("Loco/CircularWalking1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel9, root_angular_vel9, store_data9, local_matrix9, features9, bone_orientation9, phase9, bone_velocity9 = process_mocap("Loco/RegularWalking1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel10, root_angular_vel10, store_data10, local_matrix10, features10, bone_orientation10, phase10, bone_velocity10 = process_mocap("Loco/RegularWalking1.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel11, root_angular_vel11, store_data11, local_matrix11, features11, bone_orientation11, phase11, bone_velocity11 = process_mocap("Loco/WalkSideBack2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel12, root_angular_vel12, store_data12, local_matrix12, features12, bone_orientation12, phase12, bone_velocity12 = process_mocap("Loco/WalkSideBack2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel13, root_angular_vel13, store_data13, local_matrix13, features13, bone_orientation13, phase13, bone_velocity13 = process_mocap("Loco/WalkSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel14, root_angular_vel14, store_data14, local_matrix14, features14, bone_orientation14, phase14, bone_velocity14 = process_mocap("Loco/WalkSideBack3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel15, root_angular_vel15, store_data15, local_matrix15, features15, bone_orientation15, phase15, bone_velocity15 = process_mocap("Loco/WalkTurn2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel16, root_angular_vel16, store_data16, local_matrix16, features16, bone_orientation16, phase16, bone_velocity16 = process_mocap("Loco/WalkTurn2.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel17, root_angular_vel17, store_data17, local_matrix17, features17, bone_orientation17, phase17, bone_velocity17 = process_mocap("Loco/WalkTurn3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel18, root_angular_vel18, store_data18, local_matrix18, features18, bone_orientation18, phase18, bone_velocity18 = process_mocap("Loco/WalkTurn3.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)
	root_linear_vel19, root_angular_vel19, store_data19, local_matrix19, features19, bone_orientation19, phase19, bone_velocity19 = process_mocap("Idle/Idle.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	root_linear_vel20, root_angular_vel20, store_data20, local_matrix20, features20, bone_orientation20, phase20, bone_velocity20 = process_mocap("Idle/Idle.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)

	root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4, root_linear_vel5, root_linear_vel6, root_linear_vel7, root_linear_vel8, root_linear_vel9, root_linear_vel10, root_linear_vel11, root_linear_vel12, root_linear_vel13, root_linear_vel14, root_linear_vel15, root_linear_vel16, root_linear_vel17, root_linear_vel18, root_linear_vel19, root_linear_vel20))
	root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4, root_angular_vel5, root_angular_vel6, root_angular_vel7, root_angular_vel8, root_angular_vel9, root_angular_vel10, root_angular_vel11, root_angular_vel12, root_angular_vel13, root_angular_vel14, root_angular_vel15, root_angular_vel16, root_angular_vel17, root_angular_vel18, root_angular_vel19, root_angular_vel20))
	store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4, store_data5, store_data6, store_data7, store_data8, store_data9, store_data10, store_data11, store_data12, store_data13, store_data14, store_data15, store_data16,store_data17, store_data18, store_data19, store_data20))
	local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4, local_matrix5, local_matrix6, local_matrix7, local_matrix8, local_matrix9, local_matrix10, local_matrix11, local_matrix12, local_matrix13, local_matrix14, local_matrix15, local_matrix16, local_matrix17, local_matrix18, local_matrix19, local_matrix20))
	features = np.concatenate((features1, features2, features3, features4, features5, features6, features7, features8, features9, features10, features11, features12, features13, features14, features15, features16, features17, features18, features19, features20))
	bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4, bone_orientation5, bone_orientation6, bone_orientation7, bone_orientation8, bone_orientation9, bone_orientation10, bone_orientation11, bone_orientation12, bone_orientation13, bone_orientation14, bone_orientation15, bone_orientation16, bone_orientation17, bone_orientation18, bone_orientation19, bone_orientation20))
	phase = np.concatenate((phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8, phase9, phase10, phase11, phase12, phase13, phase14, phase15, phase16, phase17, phase18, phase19, phase20))
	bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4, bone_velocity5, bone_velocity6, bone_velocity7, bone_velocity8, bone_velocity9, bone_velocity10, bone_velocity11, bone_velocity12, bone_velocity13, bone_velocity14, bone_velocity15, bone_velocity16, bone_velocity17, bone_velocity18, bone_velocity19, bone_velocity20))

	# print(root_linear_vel1.shape[0], root_linear_vel2.shape[0], root_linear_vel3.shape[0], root_linear_vel4.shape[0], root_linear_vel5.shape[0], root_linear_vel6.shape[0], root_linear_vel7.shape[0], root_linear_vel8.shape[0], root_linear_vel9.shape[0], root_linear_vel10.shape[0], root_linear_vel11.shape[0], root_linear_vel12.shape[0], root_linear_vel13.shape[0], root_linear_vel14.shape[0], root_linear_vel15.shape[0], root_linear_vel16.shape[0], root_linear_vel17.shape[0], root_linear_vel18.shape[0], root_linear_vel19.shape[0], root_linear_vel20.shape[0])


	# root_linear_vel1, root_angular_vel1, store_data1, local_matrix1, features1, bone_orientation1, phase1, bone_velocity1 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=True)
	# root_linear_vel2, root_angular_vel2, store_data2, local_matrix2, features2, bone_orientation2, phase2, bone_velocity2 = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/WalkRandom.bvh/Phases_Standard.txt")
	# root_linear_vel3, root_angular_vel3, store_data3, local_matrix3, features3, bone_orientation3, phase3, bone_velocity3 = process_mocap("Idle/Idle.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt")
	# root_linear_vel4, root_angular_vel4, store_data4, local_matrix4, features4, bone_orientation4, phase4, bone_velocity4 = process_mocap("Idle/Idle.bvh", "NSM_phase2/RunSideBack3.bvh/Phases_Standard.txt", mirror=True)	
	# root_linear_vel = np.concatenate((root_linear_vel1, root_linear_vel2, root_linear_vel3, root_linear_vel4))
	# root_angular_vel = np.concatenate((root_angular_vel1, root_angular_vel2, root_angular_vel3, root_angular_vel4))
	# store_data = np.concatenate((store_data1, store_data2, store_data3, store_data4))
	# local_matrix = np.concatenate((local_matrix1, local_matrix2, local_matrix3, local_matrix4))
	# features = np.concatenate((features1, features2, features3, features4))
	# bone_orientation = np.concatenate((bone_orientation1, bone_orientation2, bone_orientation3, bone_orientation4))
	# phase = np.concatenate((phase1, phase2, phase3, phase4))
	# bone_velocity = np.concatenate((bone_velocity1, bone_velocity2, bone_velocity3, bone_velocity4))

	# root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, bone_velocity = process_mocap("Loco/WalkRandom.bvh", "NSM_phase2/RunRandom.bvh/Phases_Mirrored.txt", mirror=False)

	feature_mean = np.mean(features, axis=0)
	feature_std = np.std(features, axis=0)
	feature_mean[:] = 0
	feature_std[:] = 1
	features = (features[:-1, :] - feature_mean[:]) / feature_std[:]
	# features[:, 6:12] *= 2
	tree = KDTree(features)

	local_matrix_mean = np.mean(local_matrix, axis=0)
	local_matrix_std = np.std(local_matrix, axis=0)

	return tree, root_linear_vel, root_angular_vel, store_data, local_matrix, features, bone_orientation, phase, feature_mean, feature_std, bone_velocity

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
		self.starting_frame = 2#3185+3700+3786+4432+3527+4432+3527+3185+3786+1#1+3185+3700#7000#177#400
		self.build_dataset_for_RL()
	
	def build_dataset_for_RL(self):
		self.tree, self.root_linear_vel, self.root_angular_vel, self.store_data, self.local_matrix, self.features, self.bone_orientation, self.phase, self.feature_mean, self.feature_std, self.bone_velocity = build_KD_tree()
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
		self.next_phase = np.tile(self.phase[self.starting_frame, 0, 0:9].copy(), (self.num_envs, 1))
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

		# self.reference[:, 63:75] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, 0.13, 0.438, 0.85, -0.2, 0.35, 0.34])
		# self.reference[:, 51:63] = np.array([1.0, 0.0, 0.0, 0.0, 0.88, -0.13, -0.13, -0.438, 0.85, -0.2, -0.35, -0.34])

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
