import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()#episode_ids is just a set of indices
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)#40, 10, etc.

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]#episode_id is a thing from 0 to 49, since there are in total 50 episodes
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            #print("original_action_shape",original_action_shape)#(400, 14)#e.g. (335, 7)
            episode_len = original_action_shape[0]#400 of course
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
                #print("index",index,"start_ts",start_ts)#that is going to be the test!I want to see if same index can lead to different start_ts!
            #print("start_ts:",start_ts)#71 for example#sample 10 in the validation phase, 40 in the training phase
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]#at time start_ts
            qvel = root['/observations/qvel'][start_ts]
            #print("qpos.size,qval.size:",qpos.size,qvel.size)#14 14#7 7
            #print("qpos",qpos)
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                #print("image_dict[cam_name].size()",image_dict[cam_name].size)#921600=480*640*3! it is just one piece of image!230400=240*320*3!
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]#this : sign tells everything!
                #print("action[start_ts]", root['/action'][start_ts])
                action_len = episode_len - start_ts#
                #print("action.size,action_len",action.size,action_len)#action.size=14*action_len!#action.size=7*action_len!#
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim


        '''
        nfo=1228#951#Need to think about it#nfo means new f
        #padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action = np.zeros((nfo,original_action_shape[1]), dtype=np.float32)#
        #so padded_action's 0th dimension is 951
        padded_action[:action_len] = action#only the first action_len can be non-zero, and the remaining will be all zero
        #is_pad = np.zeros(episode_len)
        is_pad = np.zeros(nfo)
        is_pad[action_len:] = 1#after action length, everything is zero, so it is called paddled
        #print("padded_action.shape,is_pad.shape:",padded_action.shape,is_pad.shape)# (400, 14) (400,)
        '''
        numqueries=100
        padded_action = np.zeros((numqueries, original_action_shape[1]), dtype=np.float32)  #
        if(action_len<=numqueries):
            padded_action[:action_len] = action  #so padded_action will always have its first dimension to be 100!
        else:
            padded_action[:]=action[:numqueries]
        is_pad = np.zeros(numqueries)
        if(action_len<numqueries):
            is_pad[action_len:] = 1  #so is_pad always has dimension 100


        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:#in simulation there is only one camera
            all_cam_images.append(image_dict[cam_name])#
        all_cam_images = np.stack(all_cam_images, axis=0)
        #print("all_cam_images.shape:",all_cam_images.shape)# (1, 480, 640, 3)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)#now it will be (1,3,480,640)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]#now it is normalized!
        action_data = action_data#(action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = qpos_data#(qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        #print("shape: image,qpos,action,ispad:",image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
        #torch.Size([1, 3, 480, 640]),torch.Size([14]),torch.Size([400, 14]),torch.Size([400])
        return image_data, qpos_data, action_data, is_pad#no effort at all!#that is just one piece of data!


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]#it is a 2d array
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data
    #print("all_qpos_data:", all_qpos_data.shape, "all_action_data:", all_action_data.shape)
    #torch.Size([50, 400, 14]), torch.Size([50, 400, 14])
    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    #print("action_mean", action_mean.numpy().squeeze(), "action_std", action_std.numpy().squeeze(),
    #         "qpos_mean", qpos_mean.numpy().squeeze(), "qpos_std", qpos_std.numpy().squeeze())
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

def get_norm_stats_xht(dataset_dir, num_episodes):
    all_qpos_data = None#[]#np.array()#
    all_action_data = None#[]
    flagq=0
    flaga=0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]#it is a 2d numpy array
            qvel = root['/observations/qvel'][()]#it is a 2d numpy array
            action = root['/action'][()]#it is a 2d numpy array
            #qposmean = np.mean(qpos,axis=0,keepdims=True)#qpos.mean(dim=[0, 1])
            #qvelmean = np.mean(qvel,axis=0,keepdims=True)
            #actionmean=np.mean(action,axis=0,keepdims=True)#)#just a try
        if(not flagq):#(len(action.shape) == 0):
            all_qpos_data=qpos#torch.from_numpy(qpos)#.float().unsqueeze())
            flagq=1
        else:
            #aq0=all_qpos_data.pop()
            all_qpos_data=np.vstack((all_qpos_data,qpos))
        if(flaga==0):#all_action_data=:
            all_action_data=action#.append(torch.from_numpy(action))
            flaga=1
        else:
            all_action_data=np.vstack((all_action_data,action))
    #all_qpos_data = torch.stack(all_qpos_data)
    #all_action_data = torch.stack(all_action_data)
    #all_action_data = all_action_data
    #print("all_qpos_data:", all_qpos_data.shape, "all_action_data:", all_action_data.shape)#torch.Size([50, 1, 7])  torch.Size([50, 7])
    #(20000,14), (20000,14)#torch.Size([50, 400, 14]), torch.Size([50, 400, 14])
    all_qpos_data = torch.from_numpy(all_qpos_data)
    all_action_data = torch.from_numpy(all_action_data)
    # normalize action data
    #action_mean = all_action_data.mean(dim=[0,1], keepdim=True)
    #action_std = all_action_data.std(dim=[0,1], keepdim=True)
    #action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    # normalize qpos data
    #qpos_mean = all_qpos_data.mean(dim=[0,1], keepdim=True)
    #qpos_std = all_qpos_data.std(dim=[0,1], keepdim=True)
    #qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    #print("action_mean.numpy().squeeze().shape:", action_mean.numpy().squeeze().shape,"qpos_mean.numpy().squeeze()",qpos_mean.numpy().squeeze().shape)#(7,) (7,)
    #print("action_mean", action_mean.numpy().squeeze(), "action_std", action_std.numpy().squeeze(),
    #      "qpos_mean", qpos_mean.numpy().squeeze(), "qpos_std", qpos_std.numpy().squeeze())
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.81#0.85#0.8#0.8 is the default value!
    #print("num_episodes",num_episodes)#50!
    shuffled_indices = np.random.permutation(num_episodes)
    print(f'Shuffled indices: {shuffled_indices}')# in this case it is [2 1 4 0 3 5]#
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    #norm_stats = get_norm_stats(dataset_dir, num_episodes)  #
    norm_stats = get_norm_stats_xht(dataset_dir, num_episodes)#None#get_norm_stats(dataset_dir, num_episodes)#

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)#len(train_dataset)=40#
    #this is the train dataset!#
    #print("len(train_dataset): 40",len(train_dataset),"train_dataset[1]:",train_dataset[1])#
    print( "train_dataset[0/1/2/3].shape:", train_dataset[0][0].shape,train_dataset[0][1].shape,train_dataset[0][2].shape,train_dataset[0][3].shape)
    #torch.Size([2, 3, 240, 320]) torch.Size([7]) torch.Size([951, 7]) torch.Size([951])#train_dataset[1])  #
    #now this 2 refers to 2 viewpoints!
    #train_dataset's size (really?): torch.Size([1, 3, 480, 640]), torch.Size([14]), torch.Size([400, 14]), torch.Size([400])
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    #this is the validation dataset!
    print("val_dataset.shape():",len(val_dataset))#Why it is 10? because there are 50*0.2=10 episodes for validation!
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    #print(f'Train dataset size is 40: {len(train_dataset)}',"train_dataloader.size()", len(train_dataloader))#len(train_dataloader) is 5 since 40/8=5!
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])#no tilting, just regular upright
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}#all the keys
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items#kind of the average
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
