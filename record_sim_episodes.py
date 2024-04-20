import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy#this refers to the policy of making demonstrations, not the control policy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']#cube or insert or blablabla
    dataset_dir = args['dataset_dir']#
    num_episodes = args['num_episodes']#recommended 50 trajectories
    onscreen_render = args['onscreen_render']#do you want it to be shown on screen?
    inject_noise = False#
    render_cam_name = 'angle'#what is this?#is it just the camera name? Seems not

    if not os.path.isdir(dataset_dir):#if dataset_dir not existed
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']#400, correspond well with the paper!
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']#['top']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy#that is the policy for the demo, not the control policy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy#it is a scripted policy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):#50
        print(f'{episode_idx=}')#the first thing to be printed out!
        print('Rollout out EE space scripted policy')#EE space means end effector space!
        # setup the environment
        env = make_ee_sim_env(task_name)#gym like api
        ts = env.reset()#gym like api, it is like 0, right?
        episode = [ts]#already 1
        policy = policy_cls(inject_noise)#instantiation
        # setup plotting
        if onscreen_render:#if decide to show on screeen
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):#400
            action = policy(ts)#from time get the (trained) policy at that time
            ts = env.step(action)#then make one step forward, so each ts contains reward at this time step!
            episode.append(ts)#episode is a list
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()
        #print("len(episode)", len(episode))#401#
        episode_return = np.sum([ts.reward for ts in episode[1:]])#total return of this trajectory!
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])#max reward along the whole trajectory!
        if episode_max_reward == env.task.max_reward:#Now I know! env.task.max_reward is 4 as shown in the code!
            print(f"{episode_idx=} Successful, {episode_return=}")#as long as 4 is attained in at least one time step, then it will be counted as success!!!
        else:
            print(f"{episode_idx=} Failed")
        #joint_traj is just a 2d list!
        joint_traj = [ts.observation['qpos'] for ts in episode]#you get the qpos along the way from the observation member function
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]#for ts in this single episode, you get the gripper control command unnormalized
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):#they have the same size#why having this step?
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])#to show how much to open in a normalized manner
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])#to normalize it#I think this is a precaution just to make sure
            joint[6] = left_ctrl#now it is normalized! So each joint has 14 dimensions
            joint[6+7] = right_ctrl#action is 8 dim, 3pos+4quat+1gripper, while qpos/qvel are both 7dim, 6joints+1gripper! The 2nd gripper is just its mirror!

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        #I don't need this, because in the real world I don't need to provide the box position
        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()#reset the configuration with the box pose the same as the previous case

        episode_replay = [ts]#why replay?
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]#t start from 0, increase by 1 every time#it is just joint angles#the gripper pos in action is normalized
            ts = env.step(action)#this action is in the joint space
            episode_replay.append(ts)#ts is a dictionary containing all the required information of this time step
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()
        #print("len(episode_replay)",len(episode_replay))#402#
        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {#it is a dictionary. This is important for data storage!
            '/observations/qpos': [],#the value is a list
            '/observations/qvel': [],#the value is a list
            '/action': [],
        }
        for cam_name in camera_names:#top only in simulation#the value of this key is a list
            data_dict[f'/observations/images/{cam_name}'] = []#in this case it will be /observations/images/top

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]#still within an episode/one trajectory
        episode_replay = episode_replay[:-1]
        #print("len(joint_traj)",len(joint_traj))#400#
        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)#400 now
        while joint_traj:#it is the action
            action = joint_traj.pop(0)#action is a 14 dimensional list#just to get the teacher/demo joint trajectory as the action
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])#so it is a 2d list
            data_dict['/observations/qvel'].append(ts.observation['qvel'])#so it is a 2d list
            data_dict['/action'].append(action)#so it is a list of 14d things#use the qpos of the 1st play as the action/expert demonstration for the 2nd play
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            #print("action,qpos,qvel,image[top]:",action.shape,ts.observation['qpos'].shape,ts.observation['qvel'].shape,ts.observation['images']['top'].shape)
            #action, qpos, qvel, image[top]: (14,)(14, )(14, )(480, 640, 3)
            #print("action,qpos,qvel,image[top]:", type(action), type(ts.observation['qpos']),type(ts.observation['qvel']), type(ts.observation['images']['top']))
            #print("action:", action)#
            #[-1.16261004e-03 -3.72164820e-01  1.10699624e+00 -1.83124041e-03  -7.36030305e-01  1.35873051e-03  0.00000000e+00  3.13629804e-05 -2.96925582e-01  3.10684926e-01 -3.28045546e-05  1.03356532e+00  5.03137554e-05  1.00000000e+00]
            #action, qpos, qvel, image[top]: <class 'numpy.ndarray'> < class 'numpy.ndarray' > < class 'numpy.ndarray' > < class 'numpy.ndarray' >

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:#top, etc.
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))#are they really used or not?
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():#key value pairs
                #print(name,len(array))#len(array) is always 400#
                print(name, array[0].shape,type(array[0]))  #each element in array has dimension (14,) or (480,640,3), with type <class 'numpy.ndarray'>
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

