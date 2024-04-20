import rosbag
from scipy.interpolate import interp1d
import os
import tf2_ros
import rospy
#from replay_buffer import ReplayBuffer
import numpy as np
#import transformations as tf
import cv2
from cv_bridge import CvBridge
import sys
import h5py#now it should be with normalized actions and also a good choice of the supervision signal
import torch#it is in the venv!

import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
from constants_xht import DT
from constants_xht import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos, visualize_joints,visualize_differences,visualize_differences_more

from constants_xht import NUMBEROFARMS,DOFLEFT,DOFRIGHT

bag_files_directory ='/home/user/bagfilesxht2/pick_and_place_rosbag/0408_rosbag/left'##'/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag'#'/home/user/CODINGrepos/Imitation_Learning/record_data/' #sys.argv[1]
#output_file = sys.argv[1] + "/dataset"
dataset_dir=bag_files_directory+'/pick_and_place_human'#'/pick_and_place_human2'#'/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag'
task_time = 120
image_w = 320
image_h = 240

bag_topics = [
    "/camera/color/camera_info",
    "/camera/color/image_raw/compressed",
    "/camera_hand/color/camera_info",
    "/camera_hand/color/image_raw/compressed",
    "/control/motor_angle",
    "/joint_states",
    "/tf",
    "/tf_static"
]


class Data:
    def __init__(self, time):
        self.time = time
        self.img_camera = None
        self.img_camera_hand = None
        #self.tf = None
        self.vel = None
        self.last_transform = None
        self.motor_angle = None
        #self.delta_tf = None
        self.action = None#Now, the action bears the meaning of 6d joint angles+1d normalized gripper position
        self.joint_angle = None
        self.joint_angular_velocity = [0]*7#None
        self.joint_effort = [0]*7#None

class Datap(Data):
    def __init__(self,time):
        super().__init__(time)
        self.prediction = None
        self.error = None

def process_bag_file(bag_file_path, bag_topics):
    print("bag_file_path", bag_file_path)
    image_save_folder = bag_file_path + '_image'
    camera_time = None
    img_camera = None
    img_camera_hand = None
    motor_angle = None
    eff_frame_id = 'eef'
    base_frame_id = 'body'
    motor_time_buffer = []
    joint_time_buffer = []
    motor_angle_buffer = []
    joint_angle_buffer = []
    joint_angular_velocity_buffer = []
    joint_effort_buffer = []
    episode_buffer = []
    motor_interp = None
    last_transform = None
    #tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=task_time))
    image_body_buffer = []
    image_hand_buffer = []
    br = CvBridge()

    with rosbag.Bag(bag_file_path, 'r') as bag:#for me, read twice will be OK
        for topic, msg, t in bag.read_messages(topics=bag_topics):
            if topic == "/control/motor_angle":#it is a single angle!
                time = t.to_sec()
                angle = msg.data#it is a scalar
                angle=(angle-10)/(100-10)#this is the normalization step to make it between 0 and 1 inclusive
                #going to normalize it!
                motor_time_buffer.append(time)#it is a list
                motor_angle_buffer.append(angle)

            if topic == "/camera/color/image_raw/compressed":
                camera_time = msg.header.stamp.to_sec()
                data = Data(camera_time)
                img_camera = br.compressed_imgmsg_to_cv2(msg)
                data.img_camera = np.array(cv2.resize(img_camera, (image_w, image_h)))
                image_body_buffer.append(data)#so it is a list of Data objects!
            if topic == "/camera_hand/color/image_raw/compressed":  # In order to solve camera sync problem
                img_camera_hand = br.compressed_imgmsg_to_cv2(msg)
                img_camera_hand = np.array(cv2.resize(img_camera_hand, (image_w, image_h)))
                image_hand_buffer.append(img_camera_hand)
            if topic == "/joint_states":
                time = t.to_sec()
                joint_time_buffer.append(time)  # it is a list
                qpos = msg.position#data
                qvel=msg.velocity#seems no directly the velocity of the gripper
                qeffort=msg.effort
                #motor_time_buffer.append(time)#it is a list
                #print("qpos.size, qvel.size, qeffort.size:", qpos, qvel, qeffort)
                #print("qpos.size, qvel.size, qeffort.size:",len(qpos), len(qvel), len(qeffort))#6,0,0
                joint_angle_buffer.append(qpos)
                joint_angular_velocity_buffer.append(qvel)#not recorded
                joint_effort_buffer.append(qeffort)#not recorded
        motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, fill_value='extrapolate')

        joint_angle_interp = [
            interp1d(joint_time_buffer, [tup[i] for tup in joint_angle_buffer], fill_value='extrapolate') for i in
            range(len(joint_angle_buffer[0]))]

        def interpolate_joint_angles(time):
            return [interp_func(time) for interp_func in
                    joint_angle_interp]  # it is a list!#np.array([interp_func(time) for interp_func in joint_angle_interp])

        # joint_angle_interp = interp1d(joint_time_buffer, joint_angle_buffer, fill_value='extrapolate')
        joint_angular_velocity_interp = None  # interp1d(joint_time_buffer, joint_angular_velocity_buffer, fill_value='extrapolate')
        joint_effort_interp = None  # interp1d(joint_time_buffer, joint_effort_buffer, fill_value='extrapolate')

        #for idx, buffer_data in enumerate(image_body_buffer):#be accustom to the time of the camera
        for i in range(len(image_body_buffer)):
            buffer_data=image_body_buffer[i]
            camera_time = buffer_data.time#buffer_data is a Data object
            if joint_angle_interp is not None:
                joint_angle = interpolate_joint_angles(
                    camera_time)  # it is now a numpy array now#joint_angle_interp(camera_time)#get the angle at the camera time!
                # print("joint_angle", joint_angle)#6
                buffer_data.joint_angle = joint_angle
                if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angle is not None:
                    episode_buffer[len(episode_buffer) - 1].action = buffer_data.joint_angle.copy()  # .tolist()#now it has dimension 6

            if motor_interp is not None:
                motor_angle = motor_interp(camera_time)  # get the angle at the camera time!
                # print("motor_angle",motor_angle)#1
                buffer_data.motor_angle = motor_angle  # because up to now, the action is a list already!

                buffer_data.joint_angle.append(motor_angle)
                if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angle is not None:
                    episode_buffer[len(episode_buffer) - 1].action.append(motor_angle)  # now it has dimension 7

            if joint_angular_velocity_interp is not None:
                joint_angular_velocity = joint_angular_velocity_interp(camera_time)  # get the angle at the camera time!
                buffer_data.joint_angular_velocity = joint_angular_velocity
                #if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angular_velocity is not None:
                #    episode_buffer[len(episode_buffer) - 1].joint_angular_velocity = buffer_data.joint_angular_velocity.copy().tolist()  # now it has dimension 6
            if joint_effort_interp is not None:
                joint_effort = joint_effort_interp(camera_time)  # get the angle at the camera time!
                buffer_data.joint_effort = joint_effort
                #if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_effort is not None:
                    #episode_buffer[len(episode_buffer) - 1].joint_effort = buffer_data.joint_effort.copy().tolist()  # now it has dimension 6
            # deal with vel
            vel = [0, 0, 0]
            if vel is not None:
                buffer_data.vel = vel
            # deal with camera_hand
            if image_hand_buffer is not None:
                if i < len(image_hand_buffer):#if idx < len(image_hand_buffer):
                    buffer_data.img_camera_hand = image_hand_buffer[i]

            episode_buffer.append(buffer_data)

    return episode_buffer#so it is a list of Data objects

def process_bag_file_realtime_infer(bag_files_directory, bag_file, bag_topics,policy,pre_process,post_process):

    '''
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']  # not is_sim
    policy_class = config['policy_class']  # ACT or something else
    # onscreen_render = config['onscreen_render']#currently useless
    policy_config = config['policy_config']
    camera_names = config['camera_names']#['body','righthand']
    max_timesteps = config['episode_len']  # 400
    # task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    # onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)  # 121
    loading_status = policy.load_state_dict(torch.load(ckpt_path))  # loading the parameters stored in the checkpoint
    print(loading_status)
    policy.cuda()
    policy.eval()  # just to disable batch normalization and dropout
    print(f'Loaded: {ckpt_path}')  #
    stats_path = os.path.join(ckpt_dir,
                              f'dataset_stats.pkl')  # seems like this file also stores some data (of the dataset)

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']  # normalization
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']  # unnormalization

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
    '''



    def get_image_xht(image_dict, camera_names):#, t):#where to put this function is a question
        curr_images = []#put this function outside the function may save a little time?
        for cam_name in camera_names:  # ['top'] only? Yes in simulation
            curr_image = rearrange(image_dict[cam_name],#[t],
                                   'h w c -> c h w')  # don't quite know the details, but know what it means!
            curr_images.append(curr_image)  # so current_images contains images in all angles/aspects/viewpoints
        curr_image = np.stack(curr_images, axis=0)  # stack current images together
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # normalize and other processing
        return curr_image

    bag_file_path=os.path.join(bag_files_directory, bag_file)#bag_files_directory+bag_file#
    bag_time=bag_file[:-4]
    print("bag_file_path", bag_file_path)#it is a time consumer
    image_save_folder = bag_file_path + '_image'
    camera_time = None
    img_camera = None
    img_camera_hand = None
    motor_angle = None
    eff_frame_id = 'eef'
    base_frame_id = 'body'
    '''
    motor_time_buffer = []#some of them are useless in this context
    joint_time_buffer = []
    motor_angle_buffer = []
    joint_angle_buffer = []
    joint_angular_velocity_buffer = []
    joint_effort_buffer = []
    episode_buffer = []
    motor_interp = None
    last_transform = None
    #tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=task_time))
    image_body_buffer = []
    image_hand_buffer = []
    '''


    br = CvBridge()
    '''
    bodyimagecount=0
    motoranglecount=0
    handimagecount=0
    jointstatescount=0
    #predictedaction=None
    #qposgt=[]#在4/5的时候就可以弄初始的了？
    #actiongt=[]
    flagqpos=0
    flaggt=0
    flaginfer=0
    '''
    with rosbag.Bag(bag_file_path, 'r') as bag:#for me, read twice will be OK
        #print("start over!")
        motor_time_buffer = []  # some of them are useless in this context
        joint_time_buffer = []
        motor_angle_buffer = []
        joint_angle_buffer = []
        joint_angular_velocity_buffer = []
        joint_effort_buffer = []
        episode_buffer = []
        motor_interp = None
        last_transform = None
        # tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=task_time))
        image_body_buffer = []
        image_hand_buffer = []
        bodyimagecount = 0
        motoranglecount = 0
        handimagecount = 0
        jointstatescount = 0
        # predictedaction=None
        # qposgt=[]#在4/5的时候就可以弄初始的了？
        # actiongt=[]
        flagqpos = 0
        flaggt = 0
        flaginfer = 0
        flagbodycount=0
        for topic, msg, t in bag.read_messages(topics=bag_topics):
            #print('topic, t:',topic, t)
            if topic == "/control/motor_angle":#it is a single angle!
                motoranglecount+=1
                times = t.to_sec()
                angle = msg.data#it is a scalar
                angle=(angle-10)/(100-10)#this is the normalization step to make it between 0 and 1 inclusive#needs to recover after inference
                #going to normalize it!
                motor_time_buffer.append(times)#it is a list
                motor_angle_buffer.append(angle)
                #print("motoranglecount,len(motor_time_buffer),len(motor_angle_buffer):", motoranglecount,len(motor_time_buffer), len(motor_angle_buffer))
                #print("motor_time_buffer,motor_angle_buffer:",motor_time_buffer,motor_angle_buffer)
            if topic == "/camera_hand/color/image_raw/compressed":  # In order to solve camera sync problem
                handimagecount+=1
                camerahand_time = msg.header.stamp.to_sec()
                #print("camerahand_time:", camerahand_time)
                img_camera_hand = br.compressed_imgmsg_to_cv2(msg)
                img_camera_hand = np.array(cv2.resize(img_camera_hand, (image_w, image_h)))
                image_hand_buffer.append(img_camera_hand)
            if topic == "/joint_states":
                jointstatescount+=1
                times = t.to_sec()
                joint_time_buffer.append(times)  # it is a list
                qpos = msg.position#data
                qvel=msg.velocity#seems no directly the velocity of the gripper
                qeffort=msg.effort
                #motor_time_buffer.append(time)#it is a list
                #print("qpos.size, qvel.size, qeffort.size:", qpos, qvel, qeffort)
                #print("qpos.size, qvel.size, qeffort.size:",len(qpos), len(qvel), len(qeffort))#6,0,0
                joint_angle_buffer.append(qpos)
                joint_angular_velocity_buffer.append(qvel)#not recorded
                joint_effort_buffer.append(qeffort)#not recorded
            if topic == "/camera/color/image_raw/compressed":
                bodyimagecount+=1
                t0=time.time()
                camera_time = msg.header.stamp.to_sec()
                #print("camera_time:",camera_time)
                #data = Datap(camera_time)#Data(camera_time)#I think I no longer need this data!
                img_camera = br.compressed_imgmsg_to_cv2(msg)
                bodyimage=np.array(cv2.resize(img_camera, (image_w, image_h)))
                #data.img_camera = np.array(cv2.resize(img_camera, (image_w, image_h)))
                image_body_buffer.append(bodyimage)#(data)#so it is a list of Data objects!
                image_dict = dict()
                image_dict['body']=bodyimage#data.img_camera
                #image_dict[cam_namebody] = root[f'/observations/images/{cam_namebody}']  # [start_ts]
                '''
                if(bodyimagecount==5):
                    #插值获取当时的qpos
                    qposgt.append(qpos)
                    predictedaction=policy(qpos,image)
                    #actioninfer
                '''
                if(bodyimagecount>=5):#
                    #here, do the interpolation and inference
                    #bodyimagecount=0
                    #maybe timing is needed
                    #print("motor_time_buffer,motor_angle_buffer:", motor_time_buffer, motor_angle_buffer)
                    #print("bodyimagecount",bodyimagecount)
                    if(len(motor_time_buffer)<=1 or len(motor_time_buffer)<=1):
                        continue
                    #motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, fill_value='extrapolate')#for loop里面来回的创建插值对象和函数吧
                    #now I will try nearest to see if it helps! And in my case, nearest is equivalent to previous and zero
                    motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, kind='nearest',fill_value='extrapolate')  #
                    if(len(joint_angle_buffer[0])<=1 or len(joint_time_buffer)<=1):
                        continue
                    joint_angle_interp = [interp1d(joint_time_buffer, [tup[i] for tup in joint_angle_buffer], fill_value='extrapolate') for i in range(len(joint_angle_buffer[0]))]
                    if(flagbodycount==0):
                        flagbodycount=1
                        startpoint=bodyimagecount
                    def interpolate_joint_angles(time):
                        return [interp_func(time) for interp_func in joint_angle_interp]  # it is a list!#np.array([interp_func(time) for interp_func in joint_angle_interp])

                    # joint_angle_interp = interp1d(joint_time_buffer, joint_angle_buffer, fill_value='extrapolate')

                    #camera_time is not none here
                    #do interpolation to the motor angle#This requires frequently creating the interp1d function
                    #if joint_angle_interp is not None:#可能性很低很低，先免了
                    joint_angle = interpolate_joint_angles(camera_time)  # it is now a numpy array now#joint_angle_interp(camera_time)#get the angle at the camera time!
                    #else:
                    #    joint_angle=[1000]*6
                    #if motor_interp is not None:
                    motor_angle = motor_interp(camera_time)  # get the angle at the camera time!
                    #motor_angle = min(1,max(0,motor_angle))#I think if using nearest/previous/zero, this line is useless
                    #else:
                    #    motor_angle=2000
                    joint_angle.append(motor_angle)#

                    if image_hand_buffer is not None:
                        image_hand =image_hand_buffer[-1]

                        #for cam_name in camera_names:  #
                        image_dict['righthand'] = image_hand#root[f'/observations/images/{cam_namehand}']  # [start_ts]
                        #if i < len(image_hand_buffer):  # if idx < len(image_hand_buffer):
                            #buffer_data.img_camera_hand = image_hand_buffer[i]
                    #if(joint_angle[0]==1000 or joint_angle[6]==2000 or image_hand_buffer is None):
                    #    continue#no enough information to do the inference

                    #then there is enough information
                    #process the data
                    #get the previous prediction
                    #if (predictedaction==None):#predictedaction is from last time step
                    #    continue
                    #think how I can get rid of it
                    '''
                    if (predictedaction!=None):##else:
                        error=joint_angle-predictedaction#according to the current setting, joint_angle is the ground truth action
                        #you can either store the error or print the error
                    else:
                        error=joint_angle
                    '''
                    #now you update your prediction
                    #predictedaction=policy(qpos,image)
                    # print('qpos.shape',qposgt.shape,'t',t)#I should append to qpos gt one by one
                    with torch.inference_mode():  # 顾名思义，inference#is it a good practice to put this command here?
                        #qposgt.append(joint_angle)#最后再截断？
                        qpos_numpy =np.asarray(joint_angle)#qposgt[t]  # np.array(obs['qpos'])#the t here is 1,2,3,4,......
                        if (flagqpos == 0):
                            qposgt = qpos_numpy#target_qpos#
                            flagqpos = 1
                        else:
                            qposgt = np.vstack((qposgt, qpos_numpy))#target_qpos))#this is for the visualization after then step
                        #qposgt.append(qpos_numpy)#到时候去掉最后一个
                        #差一个
                        if (flaggt == 0):
                            actiongt = qpos_numpy#target_qpos#
                            flaggt = 1
                        else:
                            actiongt = np.vstack((actiongt, qpos_numpy))#target_qpos))#this is for the visualization after then step
                        #actiongt.append(joint_angle)#到时候去掉第一个
                        # print("the gripper:",qpos_numpy[6])

                        qpos = pre_process(qpos_numpy)  # normalization#qpos_numpy#because I didn't make the right preprocessing of the data!#
                        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                        # qpos_history[:, t] = qpos#I need to check the camera_names list to see what it contains
                        # curr_image = get_image(ts, camera_names)#camera_names contains all the cameras, get all images at ts
                        #I need to build the image_dict
                        curr_image = get_image_xht(image_dict, camera_names)#, t)  #我是先装手还是先装body来着？
                        # image_list
                        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  #

                        ### evaluation loop
                        if temporal_agg:  # so all_time_actions is a 400*500*state_dim thing
                            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
                        #where to put all_time_actions

                        # print("curr_image.shape:",curr_image.shape)#torch.Size([1, 2, 3, 240, 320])
                        ### query policy
                        tt=bodyimagecount-startpoint
                        t1 = time.time()
                        #print("process time:", t1 - t0)
                        if config['policy_class'] == "ACT":#all obtained from config#先保留这个接口/选项
                            #query_frequency is from config
                            if tt % query_frequency == 0:  # the action only feeds in every query_frequency steps?
                                all_actions = policy(qpos, curr_image)  # what does all_actions contain?
                            if temporal_agg:  # all_actions should have 100 in one of its dimension
                                all_time_actions[[tt], tt:tt + num_queries] = all_actions  # 100 things, right?
                                # print("all_time_actions:",len(all_time_actions),len(all_time_actions[0]),len(all_time_actions[0][0]))#400 500 14
                                actions_for_curr_step = all_time_actions[:, tt]  # num_queries is 100
                                # print("size0,size1:",len(actions_for_curr_step),len(actions_for_curr_step[0]))#400,14
                                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # it is a mask
                                # if(len(actions_populated)>0):#len(actions_populated)=400 always
                                # print("len(actions_populated),len(actions_populated[1]):",len(actions_populated))#,len(actions_populated[0]))
                                actions_for_curr_step = actions_for_curr_step[actions_populated]
                                # if(len(actions_for_curr_step)>0):#actions_for_curr_step.shape=torch.Size([1-100, 14])
                                # print("t,len(afcs),len(afcs[1]):",t, len(actions_for_curr_step),actions_for_curr_step.shape)#,len(actions_for_curr_step[0]))
                                # t,len(afcs),len(afcs[1]): 99 100 torch.Size([100, 14])#t,len(afcs),len(afcs[1]): 100 100 torch.Size([100, 14])
                                k = 0.01
                                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()  # normalize? Yeah, I think so
                                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0,
                                                                                       keepdim=True)  # kind of like convolution
                            else:
                                raw_action = all_actions[:,
                                             tt % query_frequency]  # really one piece after another! 100-100-100-100
                        elif config['policy_class'] == "CNNMLP":
                            raw_action = policy(qpos, curr_image)#no need to do temporal_aggregation
                        else:
                            raise NotImplementedError
                        ### post-process actions
                        # print("raw_action.shape",raw_action.shape)#torch.Size([1, 7])
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                        predictedaction = post_process(raw_action)  # unnormalize#raw_action#because I didn't preprocess the data. I thought I did, but I didn't#
                        #target_qpos = action  # so action has the same 量纲 with qpos#this target_qpos is the inferred one, not the ground truth one
                        #predictedaction=action
                        # then it is time to compute the error!
                        # print("target_qpos.shape",target_qpos.shape)#(7,)
                        t2 = time.time()
                        #print("infer time:",t2-t1)
                        #actiongtt = actiongt[t]  # .cpu().numpy()#the evaluation is in the previous time step, so no need to do it here this time
                        # print("actiongtt.shape",actiongtt.shape)#(7,)
                        # print(f'diffactionqpos:{actiongtt-qpos_numpy},\nqpos_numpy:{qpos_numpy},\nactiongt:{actiongtt},\ninferred:{target_qpos},\ndiff:{actiongtt-target_qpos}\n')
                        if (flaginfer == 0):
                            actioninfer = predictedaction#target_qpos#
                            flaginfer = 1
                        else:
                            actioninfer = np.vstack((actioninfer, predictedaction))#target_qpos))#this is for the visualization after then step
                        ### step the environment
                        # ts = env.step(target_qpos)

                        ### for visualization
                        #qpos_list.append(qpos_numpy)#
                        #target_qpos_list.append(predictedaction)#(target_qpos)#
                        # rewards.append(ts.reward)
        #print("episode_id:", episode_id)#now use time as the identifier
        # visualize_differences(actiongt, actioninfer,plot_path=os.path.join(dataset_dir,f'episode_{episode_id}_qpos_{teornot}_verify.png'))
        qposgt=qposgt[:-1]
        actiongt=actiongt[1:]
        actioninfer=actioninfer[:-1]
        visualize_differences_more(qposgt, actiongt, actioninfer,
                                   plot_path=os.path.join(dataset_dir,
                                                          f'qpos_{teornot}_verify_{bag_time}.png'))
                    # plt.close()
                    #THEN IT IS THOSE TE THINGS
                #better also do interpolation to the joint states
                #if there is no enough points,

    return #now you return something else, probably the array of errors#episode_buffer#so it is a list of Data objects

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)#see policy.py!
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

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

    args=vars(parser.parse_args())
    #args = parser.parse_args()



    rospy.init_node('verify_rosbag', anonymous=True)

    #if not os.path.exists(output_file):
    #    os.makedirs(output_file)
    #the main from imitate_episodes_xht goes here!
    #then you load the configuration and the policy
    #replay_buffer = ReplayBuffer.create_from_path(output_file, mode='a')

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
    ckpt_name = f'policy_best.ckpt'
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

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']  # normalization
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']  # unnormalization

    query_frequency = policy_config[
        'num_queries']  # 1 for CNNMLP, 100 for ACT, if not temporal_agg then open loop 100, otherwise temporal_agg
    if temporal_agg:  # since this is better, I will choose this?
        query_frequency = 1  # temporal agg only matters in evaluation!
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    # is max_timesteps really used?
    num_rollouts = 50  # or 51 when using 04_07?
    # episode_returns = []
    # highest_rewards = []

    if temporal_agg:  # I make this change to add teornot
        teornot = 'te'
    else:
        teornot = 'non-te'

    maxlength=0
    episode_idx = 0#2
    for bag_file in os.listdir(bag_files_directory):
        if bag_file.endswith(".bag"):
            count=0
            #episode_data = process_bag_file(os.path.join(bag_files_directory, bag_file), bag_topics)
            #episode_data = process_bag_file_realtime_infer(os.path.join(bag_files_directory, bag_file), bag_topics,policy,pre_process,post_process)
            process_bag_file_realtime_infer(bag_files_directory, bag_file, bag_topics,policy, pre_process, post_process)
            #buffer = []#Now I know that episode_data is a list of Data objects

