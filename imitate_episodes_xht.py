import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants_xht import DT
from constants_xht import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos, visualize_joints,visualize_differences,visualize_differences_more

from sim_env import BOX_POSE
from constants_xht import NUMBEROFARMS,DOFLEFT,DOFRIGHT
import IPython
e = IPython.embed
import h5py
import time
def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']#eval or train
    ckpt_dir = args['ckpt_dir']#
    policy_class = args['policy_class']#ACT or other baseline methods!
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']#cube transfer, insertion, etc.
    batch_size_train = args['batch_size']#8 default
    batch_size_val = args['batch_size']#the val batchsize is the same as the train batch size
    num_epochs = args['num_epochs']#2000 default

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        #from constants import SIM_TASK_CONFIGS
        from constants_xht import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']#50 default
    episode_len = task_config['episode_len']#400 default
    camera_names = task_config['camera_names']#it should be a list of camera_names#it seems that in simulation, it only contain one viewpoint, which is from the top!

    # fixed parameters
    if NUMBEROFARMS==2:
        state_dim=DOFRIGHT+DOFLEFT
    elif NUMBEROFARMS==1:
        state_dim=DOFRIGHT
    else:
        state_dim=14
    #state_dim = 14#(6 joints+1 gripper)*2
    lr_backbone = 1e-5#so it is not a hyperparameter?
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8#what is this hyperparameter? it is a transformer thing
        policy_config = {'lr': args['lr'],#so lr is not lr_backbone
                         'num_queries': args['chunk_size'],#100
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],#3200 by default
                         'lr_backbone': lr_backbone,#lr_backbone means learning rate for the backbone
                         'backbone': backbone,
                         'enc_layers': enc_layers,#that is not modifyable
                         'dec_layers': dec_layers,#not modifyable
                         'nheads': nheads,#not modifyable
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':#
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,#there is no action chunking in CNNMLP
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,#14 as shown in the paper
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],#is it a val thing or is it also a train thing?
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']#at this time, you can try to do simulation in mujoco
        results = []
        for ckpt_name in ckpt_names:
            #success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)#151, in this function lots of things are printed
            eval_bc_xht_hdf5(config, ckpt_name, save_episode=True,dataset_dir=dataset_dir)  #
            #results.append([ckpt_name, success_rate, avg_return])

        #for ckpt_name, success_rate, avg_return in results:
            #print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()#something like return

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)
    #dataset_dir: selfcollecteddata/sim_transfer_cube_scripted
    print("dataset_dir",dataset_dir)
    #print("length of train_/val_dataloader:",len(train_dataloader.dataset),len(val_dataloader.dataset))#40,10#I need to know the data size within the train_/val_dataloader
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)#create checkpoint directory
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')#a file storing statistics?
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)#this is the train code!
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info#

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')#
    torch.save(best_state_dict, ckpt_path)#storing best parameters at the ckpt_path
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)#see policy.py!
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:#['top'] only? Yes in simulation
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')#don't quite know the details, but know what it means!
        curr_images.append(curr_image)#so current_images contains images in all angles/aspects/viewpoints
    curr_image = np.stack(curr_images, axis=0)#stack current images together
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)#normalize and other processing
    return curr_image

def get_image_xht(image_dict, camera_names,t):
    curr_images = []
    for cam_name in camera_names:#['top'] only? Yes in simulation
        curr_image = rearrange(image_dict[cam_name][t], 'h w c -> c h w')#don't quite know the details, but know what it means!
        curr_images.append(curr_image)#so current_images contains images in all angles/aspects/viewpoints
    curr_image = np.stack(curr_images, axis=0)#stack current images together
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)#normalize and other processing
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']#not is_sim
    policy_class = config['policy_class']#ACT or something else
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']#400
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)#121
    loading_status = policy.load_state_dict(torch.load(ckpt_path))#loading the parameters stored in the checkpoint
    print(loading_status)
    policy.cuda()
    policy.eval()#just to disable batch normalization and dropout
    print(f'Loaded: {ckpt_path}')#
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')#seems like this file also stores some data (of the dataset)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']#normalization
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']#unnormalization

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)#if time permits, read this function
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)#it is not playing rosbag!
        env_max_reward = env.task.max_reward#4

    query_frequency = policy_config['num_queries']#1 for CNNMLP, 100 for ACT
    if temporal_agg:#since this is better, I will choose this?
        query_frequency = 1#temporal agg only matters in evaluation!
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    if temporal_agg:#I make this change to add teornot
        teornot='te'#temporal ensembling
    else:
        teornot='non-te'#non temporal ensembling

    for rollout_id in range(num_rollouts):#50
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:#so all_time_actions is a 400*500*state_dim thing
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()#state_dim is 14
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():#顾名思义，inference
            for t in range(max_timesteps):#0-399
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation#so obs contains 3 images in one single frame!
                if 'images' in obs:
                    #print("case1!")#it is case1!
                    image_list.append(obs['images'])
                    #print("obs['images']",obs['images'])
                else:
                    #print("case2!")
                    image_list.append({'main': obs['image']})#which case it enters?
                #print("image_list",image_list)#image_list contains 'angle','top','vis'
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)#normalization
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos#I need to check the camera_names list to see what it contains
                curr_image = get_image(ts, camera_names)#camera_names contains all the cameras, get all images at ts

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:#the action only feeds in every query_frequency steps?
                        all_actions = policy(qpos, curr_image)#what does all_actions contain?
                    if temporal_agg:#all_actions should have 100 in one of its dimension
                        all_time_actions[[t], t:t+num_queries] = all_actions#100 things, right?
                        #print("all_time_actions:",len(all_time_actions),len(all_time_actions[0]),len(all_time_actions[0][0]))#400 500 14
                        actions_for_curr_step = all_time_actions[:, t]#num_queries is 100
                        #print("size0,size1:",len(actions_for_curr_step),len(actions_for_curr_step[0]))#400,14
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)#it is a mask
                        #if(len(actions_populated)>0):#len(actions_populated)=400 always
                            #print("len(actions_populated),len(actions_populated[1]):",len(actions_populated))#,len(actions_populated[0]))
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        #if(len(actions_for_curr_step)>0):#actions_for_curr_step.shape=torch.Size([1-100, 14])
                        #print("t,len(afcs),len(afcs[1]):",t, len(actions_for_curr_step),actions_for_curr_step.shape)#,len(actions_for_curr_step[0]))
                        #t,len(afcs),len(afcs[1]): 99 100 torch.Size([100, 14])#t,len(afcs),len(afcs[1]): 100 100 torch.Size([100, 14])
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()#normalize? Yeah, I think so
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)#kind of like convolution
                    else:
                        raw_action = all_actions[:, t % query_frequency]#really one piece after another! 100-100-100-100
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)#unnormalize
                target_qpos = action#so action has the same 量纲 with qpos

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)#just this naive way of determining success!
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, teornot, f'video{rollout_id}.mp4'))  # I make this change to save directly to a folder
            #if not temporal_agg:
                #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir,'non-te', f'video{rollout_id}.mp4'))#I make this change to save directly to a folder
            #elif temporal_agg:
                #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, 'te',f'video{rollout_id}.mp4'))  # I make this change to save directly to a folder

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()#just highest_rewards in each episode!
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'


    with open(os.path.join(ckpt_dir,teornot, result_file_name), 'w') as f:#I make this change to add teornot
        f.write(summary_str)
        f.write(repr(episode_returns))#some 600 things
        f.write('\n\n')
        f.write(repr(highest_rewards))#0,1,2,3,4

    return success_rate, avg_return


def forward_pass(data, policy):#return the forward pass of the policy network
    image_data, qpos_data, action_data, is_pad = data
    #print("image shape",image_data.size(), "qpos shape:",qpos_data.size(),"action_data shape:", action_data.size(),"is_pad shape:", is_pad.size())
    #torch.Size([8, 1, 3, 480, 640]), torch.Size([8, 14]), torch.Size([8, 400, 14]),torch.Size([8, 400])#8 is during training, 2 is during validation
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    #qpos 8*14, image 8*1*3*480*640, action 8*400*14, is_pad 8*400
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None
    #I think it is calling the forward method#in evaluation mode, you just return a_hat

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']#2000 is the number of training epochs!
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']#ACT or CNNMLP?
    policy_config = config['policy_config']#those bunch of things!

    set_seed(seed)

    start_epoch = 0#2000  #5000#0 means no previous training records! Otherwise there is!
    if start_epoch == 0:
        ckpt_name=None
    else:
        ckpt_name='policy_last.ckpt'#None
    if(ckpt_name!=None):
        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy = make_policy(policy_class, policy_config)  # 121
        loading_status = policy.load_state_dict(torch.load(ckpt_path))  # loading the parameters stored in the checkpoint
        print(loading_status)
        policy.cuda()
        #policy.eval()  # just to disable batch normalization and dropout
        print(f'Loaded: {ckpt_path}')  #

    else:
        policy = make_policy(policy_class, policy_config)  #
        policy.cuda()
    optimizer = make_optimizer(policy_class, policy)  # just returning the optimizer


    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(start_epoch,num_epochs)):#2000#for epoch in tqdm(range(num_epochs)):#2000
        #print(f'\nEpoch {epoch}')#
        # validation
        with torch.inference_mode():#just the validation phase
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):#batch size is from 0 to 1, only 2 times
                #print('batch_idx: ', batch_idx,"val data length:",len(data))#only 4, since#8+2
                #print("data 0 size:",data[0].size(),"data 1.size():",data[1].size(),"data 2.size():",data[2].size(),"data 3.size():",data[3].size())
                forward_dict = forward_pass(data, policy)#I need to know the size of data, in whatever ways
                epoch_dicts.append(forward_dict)#the above command calls __getitem__ 10 times!
            epoch_summary = compute_dict_mean(epoch_dicts)#I need to read compute_dict_mean function
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        #print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        #print(summary_string)
        #print(f'Val loss:   {epoch_val_loss:.5f}',summary_string)
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):#batch size is from 0 to 4
            #print('batch_idx: ', batch_idx, "train data length:",len(data))#4
            #print("data 0.size():", data[0].size(), "data 1.size():", data[1].size(), "data 2.size():", data[2].size(),
            #      "data 3.size():", data[3].size())#now you get it!
            #data 0 size: torch.Size([8, 1, 3, 480, 640]) data 1.size(): torch.Size([8, 14]) data 2.size(): torch.Size([8, 400, 14]) data 3.size(): torch.Size([8, 400])
            forward_dict = forward_pass(data, policy)#what does the forward method return?#calls the __call__ method in policy
            # the above command calls __getitem__ 40 times!
            # backward
            loss = forward_dict['loss']
            loss.backward()#back propagation
            optimizer.step()#one step down the gradient
            optimizer.zero_grad()#which comes first?
            train_history.append(detach_dict(forward_dict))
        #epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * (epoch-start_epoch):(batch_idx + 1) * (epoch-start_epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        #print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        #print(f'Train loss: {epoch_train_loss:.5f}',summary_string)
        print(f'Val loss:   {epoch_val_loss:.5f}', summary_string,f'Train loss: {epoch_train_loss:.5f}',summary_string)#I make this change
        if epoch % 1000 == 0:#now make it 1000/500#epoch % 100 == 0:#gonna save check points every 100 epochs
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info#it is just a tuple


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

def eval_bc_xht_hdf5(config, ckpt_name, save_episode=True,dataset_dir=None):#dataset_dir or config?
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']#not is_sim
    policy_class = config['policy_class']#ACT or something else
    #onscreen_render = config['onscreen_render']#currently useless
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']#400
    #task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    #onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)#121
    loading_status = policy.load_state_dict(torch.load(ckpt_path))#loading the parameters stored in the checkpoint
    print(loading_status)
    policy.cuda()
    policy.eval()#just to disable batch normalization and dropout
    print(f'Loaded: {ckpt_path}')#
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')#seems like this file also stores some data (of the dataset)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']#normalization
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']#unnormalization
    '''
    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)#if time permits, read this function
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)#it is not playing rosbag!
        env_max_reward = env.task.max_reward#4
    '''
    query_frequency = policy_config['num_queries']#1 for CNNMLP, 100 for ACT, if not temporal_agg then open loop 100, otherwise temporal_agg
    if temporal_agg:#since this is better, I will choose this?
        query_frequency = 1#temporal agg only matters in evaluation!
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    #is max_timesteps really used?
    num_rollouts = 50#or 51 when using 04_07?
    #episode_returns = []
    #highest_rewards = []

    if temporal_agg:#I make this change to add teornot
        teornot='te'
    else:
        teornot='non-te'

    for rollout_id in range(num_rollouts):#50#range(4,5):#here I need to change it to first the hdf5 files, then the rosbag files
        #rollout_id += 0

        #sample_full_episode = False  # hardcode
        #print("rollout_id:",rollout_id)
        episode_id = rollout_id#self.episode_ids[index]  # episode_id is a thing from 0 to 49, since there are in total 50 episodes
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        print(f'dataset_path:{dataset_path}')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']#is it useless? hold it maybe
            original_action_shape = root['/action'].shape
            #print("original_action_shape",original_action_shape)#(400, 14)#e.g. (335, 7)
            episode_len = original_action_shape[0]#400 of course, not necessarily in our robot, the real length of this trajectory
            #print("start_ts:",start_ts)#71 for example#sample 10 in the validation phase, 40 in the training phase
            # get observation at start_ts only
            qposgt = root['/observations/qpos']#[start_ts]#at time start_ts#e.g. (335,7) if no [start_ts]
            qvelgt = root['/observations/qvel']#[start_ts]#e.g. (335,7)#改成rosbag的时候只用改这些就好了
            #print("qpos.size,qval.size:",qpos.shape,qvel.shape)#14 14#7 7#(503,7)
            #print("qpos",qpos)
            image_dict = dict()
            for cam_name in camera_names:#
                image_dict[cam_name] = root[f'/observations/images/{cam_name}']#[start_ts]
                #print("image_dict[cam_name].size()",image_dict[cam_name].size)#921600=480*640*3! it is just one piece of image!230400=240*320*3!
            # get all actions after and including start_ts
            if is_sim:
                actiongt = root['/action']#[start_ts:]#this : sign tells everything!
                #print("actiongt", root['/action'])#shape (503, 7)
                #action_len = episode_len - start_ts#
                #print("action.size,action_len",action.size,action_len)#action.size=14*action_len!#action.size=7*action_len!#
            else:
                actiongt = root['/action']#[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                #action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
            flaginfer=0
            #now, give you qpos, you need to get the action!
            #print("qpos,qval:", qpos, qvel)  #
            #print("qpos.size,qval.size:", qpos.size, qvel.size)  # 14 14#7 7
            #I can do a visualization by following the visualization_episode code!
            ### evaluation loop
            if temporal_agg:#so all_time_actions is a 400*500*state_dim thing
                #all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
                all_time_actions = torch.zeros([num_queries, num_queries, state_dim]).cuda()  # just once!
                #during evaluation, max_timesteps can be episode_len
            #qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()#state_dim is 14
            image_list = [] # for visualization
            qpos_list = []
            target_qpos_list = []
            #rewards = []#NOT USED WHEN USING XHT ROBOTS
            with torch.inference_mode():#顾名思义，inference
                for t in range(episode_len):#(max_timesteps):#0-399#in this case it will first be 951#335, 951, etc.
                    ### update onscreen render and wait for DT
                    #if onscreen_render:
                    #    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    #    plt_img.set_data(image)
                    #    plt.pause(DT)
                    '''
                    ### process previous timestep to get qpos and image_list
                    obs = ts.observation#so obs contains 3 images in one single frame!
                    if 'images' in obs:
                        #print("case1!")#it is case1!
                        image_list.append(obs['images'])
                        #print("obs['images']",obs['images'])
                    else:
                        #print("case2!")
                        image_list.append({'main': obs['image']})#which case it enters?
                    #print("image_list",image_list)#image_list contains 'angle','top','vis'
                    '''
                    t0=time.time()
                    #print('qpos.shape',qposgt.shape,'t',t)
                    qpos_numpy = qposgt[t]#np.array(obs['qpos'])#the t here is 1,2,3,4,......
                    #print("the gripper:",qpos_numpy[6])
                    qpos = pre_process(qpos_numpy)#normalization#qpos_numpy#because I didn't make the right preprocessing of the data!#
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    #qpos_history[:, t] = qpos#I need to check the camera_names list to see what it contains
                    #curr_image = get_image(ts, camera_names)#camera_names contains all the cameras, get all images at ts
                    curr_image = get_image_xht(image_dict, camera_names,t)  #
                    #image_list
                    #curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  #
                    '''
                    for cam_name in camera_names:#
                        
                        camimaget=torch.from_numpy(image_dict[cam_name][t]/255.0).float().cuda().unsqueeze(0)
                        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')  #
                        image_list.append(camimaget)
                    if(len(image_list)):
                        curr_image=torch.cat(image_list,dim=0)
                    '''
                    #print("curr_image.shape:",curr_image.shape)#torch.Size([1, 2, 3, 240, 320])
                    ### query policy
                    t1=time.time()
                    print("process time:", t1 - t0)
                    if config['policy_class'] == "ACT":

                        if t % query_frequency == 0:#the action only feeds in every query_frequency steps?
                            all_actions = policy(qpos, curr_image)#what does all_actions contain?
                        t3=time.time()
                        print("query time:", t3-t1)
                        if temporal_agg:#all_actions should have 100 in one of its dimension
                            #all_time_actions[[t], t:t+num_queries] = all_actions#100 things, right?
                            all_time_actions[t % num_queries] = all_actions  # that line is overwritten!
                            #print("all_time_actions:",len(all_time_actions),len(all_time_actions[0]),len(all_time_actions[0][0]))#400 500 14
                            if (t >= num_queries - 1):
                                rowindex = torch.arange(num_queries)
                                columnindex = (torch.arange(t, t - 100, -1)) % num_queries
                            else:
                                rowindex = torch.arange(t + 1)
                                columnindex = torch.arange(t, -1, -1)
                            #actions_for_curr_step = all_time_actions[:, t]#num_queries is 100
                            actions_for_curr_step = all_time_actions[rowindex, columnindex]  # num_queries is 100
                            #print("size0,size1:",len(actions_for_curr_step),len(actions_for_curr_step[0]))#400,14
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)#it is a mask
                            #if(len(actions_populated)>0):#len(actions_populated)=400 always
                                #print("len(actions_populated),len(actions_populated[1]):",len(actions_populated))#,len(actions_populated[0]))
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            #if(len(actions_for_curr_step)>0):#actions_for_curr_step.shape=torch.Size([1-100, 14])
                            #print("t,len(afcs),len(afcs[1]):",t, len(actions_for_curr_step),actions_for_curr_step.shape)#,len(actions_for_curr_step[0]))
                            #t,len(afcs),len(afcs[1]): 99 100 torch.Size([100, 14])#t,len(afcs),len(afcs[1]): 100 100 torch.Size([100, 14])
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()#normalize? Yeah, I think so
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)#kind of like convolution
                        else:
                            raw_action = all_actions[:, t % query_frequency]#really one piece after another! 100-100-100-100
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    #print("raw_action.shape",raw_action.shape)#torch.Size([1, 7])
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)#unnormalize#raw_action#because I didn't preprocess the data. I thought I did, but I didn't#
                    target_qpos = action#so action has the same 量纲 with qpos#this target_qpos is the inferred one, not the ground truth one
                    #then it is time to compute the error!
                    #print("target_qpos.shape",target_qpos.shape)#(7,)

                    actiongtt=actiongt[t]#.cpu().numpy()
                    #print("actiongtt.shape",actiongtt.shape)#(7,)
                    #print(f'diffactionqpos:{actiongtt-qpos_numpy},\nqpos_numpy:{qpos_numpy},\nactiongt:{actiongtt},\ninferred:{target_qpos},\ndiff:{actiongtt-target_qpos}\n')
                    if(flaginfer==0):
                        actioninfer=target_qpos
                        flaginfer=1
                    else:
                        actioninfer=np.vstack((actioninfer,target_qpos))
                    ### step the environment
                    #ts = env.step(target_qpos)

                    ### for visualization
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    t2=time.time()
                    print("inference time:",t2-t1)
                    #rewards.append(ts.reward)
            print("episode_id:",episode_id)
            #visualize_differences(actiongt, actioninfer,plot_path=os.path.join(dataset_dir,f'episode_{episode_id}_qpos_{teornot}_verify.png'))
            visualize_differences_more(qposgt,actiongt, actioninfer,
                                      plot_path=os.path.join(dataset_dir, f'episode_{episode_id}_qpos_{teornot}_verify.png'))
                #plt.close()
            #if real_robot:
                #move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
                #pass
        '''
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)#just this naive way of determining success!
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, teornot, f'video{rollout_id}.mp4'))  # I make this change to save directly to a folder
            #if not temporal_agg:
                #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir,'non-te', f'video{rollout_id}.mp4'))#I make this change to save directly to a folder
            #elif temporal_agg:
                #save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, 'te',f'video{rollout_id}.mp4'))  # I make this change to save directly to a folder
        '''

    '''    
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()#just highest_rewards in each episode!
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    '''

    #with open(os.path.join(ckpt_dir,teornot, result_file_name), 'w') as f:#I make this change to add teornot
        #f.write(summary_str)
        #f.write(repr(episode_returns))#some 600 things
        #f.write('\n\n')
        #f.write(repr(highest_rewards))#0,1,2,3,4

    #return success_rate, avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
