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

from constants_xht import DT
from constants_xht import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos, visualize_joints,visualize_differences

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
    '''
    def convert_tf_to_list(self, tf):#the action in diffusion policy is the tranform!
        list_tf = [tf.transform.translation.x, tf.transform.translation.y,
                   tf.transform.translation.z,
                   tf.transform.rotation.x, tf.transform.rotation.y,
                   tf.transform.rotation.z, tf.transform.rotation.w]
        return list_tf#so each action is a list! which needs stacking to make it a numpy array!
    
    def delta_tf_func(self):
        l_translation = np.array(
            [self.last_transform.transform.translation.x, self.last_transform.transform.translation.y,
             self.last_transform.transform.translation.z])
        l_rotation = np.array([self.last_transform.transform.rotation.x, self.last_transform.transform.rotation.y,
                               self.last_transform.transform.rotation.z, self.last_transform.transform.rotation.w])
        last_rotation = tf.quaternion_matrix(l_rotation)
        last_translation = tf.translation_matrix(l_translation)
        last_transform_matrix = last_rotation @ last_translation
        c_translation = np.array(
            [self.tf.transform.translation.x, self.tf.transform.translation.y, self.tf.transform.translation.z])
        c_rotation = np.array([self.tf.transform.rotation.x, self.tf.transform.rotation.y, self.tf.transform.rotation.z,
                               self.tf.transform.rotation.w])
        current_rotation = tf.quaternion_matrix(c_rotation)
        current_translation = tf.translation_matrix(c_translation)
        current_transform_matrix = current_rotation @ current_translation
        delta_transform = tf.inverse_matrix(last_transform_matrix) @ current_transform_matrix
        delta_translation = tf.translation_from_matrix(delta_transform)
        delta_rotation = tf.quaternion_from_matrix(delta_transform)
        self.delta_tf = np.hstack([delta_translation, delta_rotation])
    '''

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
            '''
            if topic == "/tf_static":
                tf_msg = msg.transforms
                for transform in tf_msg:
                    tf_buffer.set_transform_static(transform, "")
            if topic == "/tf":
                tf_msg = msg.transforms
                for transform in tf_msg:
                    tf_buffer.set_transform(transform, "")
            '''
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

def process_bag_file_realtime_infer(bag_file_path, bag_topics):
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
    bodyimagecount=0
    motoranglecount=0
    handimagecount=0
    jointstatescount=0
    predictedaction=None
    with rosbag.Bag(bag_file_path, 'r') as bag:#for me, read twice will be OK
        for topic, msg, t in bag.read_messages(topics=bag_topics):
            if topic == "/control/motor_angle":#it is a single angle!
                motoranglecount+=1
                time = t.to_sec()
                angle = msg.data#it is a scalar
                angle=(angle-10)/(100-10)#this is the normalization step to make it between 0 and 1 inclusive#needs to recover after inference
                #going to normalize it!
                motor_time_buffer.append(time)#it is a list
                motor_angle_buffer.append(angle)
            '''
            if topic == "/tf_static":
                tf_msg = msg.transforms
                for transform in tf_msg:
                    tf_buffer.set_transform_static(transform, "")
            if topic == "/tf":
                tf_msg = msg.transforms
                for transform in tf_msg:
                    tf_buffer.set_transform(transform, "")
            '''
            if topic == "/camera_hand/color/image_raw/compressed":  # In order to solve camera sync problem
                handimagecount+=1
                camerahand_time = msg.header.stamp.to_sec()
                #print("camerahand_time:", camerahand_time)
                img_camera_hand = br.compressed_imgmsg_to_cv2(msg)
                img_camera_hand = np.array(cv2.resize(img_camera_hand, (image_w, image_h)))
                image_hand_buffer.append(img_camera_hand)
            if topic == "/joint_states":
                jointstatescount+=1
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
            if topic == "/camera/color/image_raw/compressed":
                bodyimagecount+=1

                camera_time = msg.header.stamp.to_sec()
                #print("camera_time:",camera_time)
                data = Datap(camera_time)#Data(camera_time)
                img_camera = br.compressed_imgmsg_to_cv2(msg)
                data.img_camera = np.array(cv2.resize(img_camera, (image_w, image_h)))
                image_body_buffer.append(data)#so it is a list of Data objects!
                if(bodyimagecount>=5):#
                    #here, do the interpolation and inference
                    #bodyimagecount=0
                    motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, fill_value='extrapolate')#for loop里面来回的创建插值对象和函数吧
                    joint_angle_interp = [interp1d(joint_time_buffer, [tup[i] for tup in joint_angle_buffer], fill_value='extrapolate') for i in range(len(joint_angle_buffer[0]))]

                    def interpolate_joint_angles(time):
                        return [interp_func(time) for interp_func in joint_angle_interp]  # it is a list!#np.array([interp_func(time) for interp_func in joint_angle_interp])

                    # joint_angle_interp = interp1d(joint_time_buffer, joint_angle_buffer, fill_value='extrapolate')
                    joint_angular_velocity_interp = None  # interp1d(joint_time_buffer, joint_angular_velocity_buffer, fill_value='extrapolate')
                    joint_effort_interp = None  # interp1d(joint_time_buffer, joint_effort_buffer, fill_value='extrapolate')

                    #camera_time is not none here
                    #do interpolation to the motor angle#This requires frequently creating the interp1d function
                    if joint_angle_interp is not None:
                        joint_angle = interpolate_joint_angles(camera_time)  # it is now a numpy array now#joint_angle_interp(camera_time)#get the angle at the camera time!
                    else:
                        joint_angle=[1000]*6
                    if motor_interp is not None:
                        motor_angle = motor_interp(camera_time)  # get the angle at the camera time!
                    else:
                        motor_angle=2000
                    joint_angle.append(motor_angle)

                    # print("joint_angle", joint_angle)#6
                    #buffer_data.joint_angle = joint_angle#seems that I no longer need this step!
                    #if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angle is not None:
                    #    episode_buffer[len(episode_buffer) - 1].action = buffer_data.joint_angle.copy()  # .tolist()#now it has dimension 6
                    if joint_angular_velocity_interp is not None:
                        joint_angular_velocity = joint_angular_velocity_interp(camera_time)  # get the angle at the camera time!
                        #buffer_data.joint_angular_velocity = joint_angular_velocity
                        # if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angular_velocity is not None:
                        #    episode_buffer[len(episode_buffer) - 1].joint_angular_velocity = buffer_data.joint_angular_velocity.copy().tolist()  # now it has dimension 6
                    if joint_effort_interp is not None:
                        joint_effort = joint_effort_interp(camera_time)  # get the angle at the camera time!
                        #buffer_data.joint_effort = joint_effort
                    if image_hand_buffer is not None:
                        image_hand =image_hand_buffer[-1]
                        #if i < len(image_hand_buffer):  # if idx < len(image_hand_buffer):
                            #buffer_data.img_camera_hand = image_hand_buffer[i]
                    if(joint_angle[0]==1000 or joint_angle[6]==2000 or image_hand_buffer is None):
                        continue#no enough information to do the inference

                    #then there is enough information
                    #process the data
                    #get the previous prediction
                    if (predictedaction==None):
                        continue
                    else:
                        error=joint_angle-predictedaction
                        #you can either store the error or print the error

                    #now you update your prediction
                    #predictedaction=policy(qpos,image)
                    # print('qpos.shape',qposgt.shape,'t',t)
                    qpos_numpy = joint_angle#qposgt[t]  # np.array(obs['qpos'])#the t here is 1,2,3,4,......
                    # print("the gripper:",qpos_numpy[6])
                    qpos = pre_process(
                        qpos_numpy)  # normalization#qpos_numpy#because I didn't make the right preprocessing of the data!#
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    # qpos_history[:, t] = qpos#I need to check the camera_names list to see what it contains
                    # curr_image = get_image(ts, camera_names)#camera_names contains all the cameras, get all images at ts
                    #I need to build the image_dict
                    curr_image = get_image_xht(image_dict, camera_names, t)  #
                    # image_list
                    # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  #
                    '''
                    for cam_name in camera_names:#

                        camimaget=torch.from_numpy(image_dict[cam_name][t]/255.0).float().cuda().unsqueeze(0)
                        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')  #
                        image_list.append(camimaget)
                    if(len(image_list)):
                        curr_image=torch.cat(image_list,dim=0)
                    '''
                    # print("curr_image.shape:",curr_image.shape)#torch.Size([1, 2, 3, 240, 320])
                    ### query policy
                    if config['policy_class'] == "ACT":#all obtained from config
                        if t % query_frequency == 0:  # the action only feeds in every query_frequency steps?
                            all_actions = policy(qpos, curr_image)  # what does all_actions contain?
                        if temporal_agg:  # all_actions should have 100 in one of its dimension
                            all_time_actions[[t], t:t + num_queries] = all_actions  # 100 things, right?
                            # print("all_time_actions:",len(all_time_actions),len(all_time_actions[0]),len(all_time_actions[0][0]))#400 500 14
                            actions_for_curr_step = all_time_actions[:, t]  # num_queries is 100
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
                                         t % query_frequency]  # really one piece after another! 100-100-100-100
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)#no need to do temporal_aggregation
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    # print("raw_action.shape",raw_action.shape)#torch.Size([1, 7])
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(
                        raw_action)  # unnormalize#raw_action#because I didn't preprocess the data. I thought I did, but I didn't#
                    target_qpos = action  # so action has the same 量纲 with qpos#this target_qpos is the inferred one, not the ground truth one
                    # then it is time to compute the error!
                    # print("target_qpos.shape",target_qpos.shape)#(7,)

                    #actiongtt = actiongt[t]  # .cpu().numpy()#the evaluation is in the previous time step, so no need to do it here this time
                    # print("actiongtt.shape",actiongtt.shape)#(7,)
                    # print(f'diffactionqpos:{actiongtt-qpos_numpy},\nqpos_numpy:{qpos_numpy},\nactiongt:{actiongtt},\ninferred:{target_qpos},\ndiff:{actiongtt-target_qpos}\n')
                    if (flaginfer == 0):
                        actioninfer = target_qpos
                        flaginfer = 1
                    else:
                        actioninfer = np.vstack((actioninfer, target_qpos))#this is for the visualization after then step
                    ### step the environment
                    # ts = env.step(target_qpos)

                    ### for visualization
                    qpos_list.append(qpos_numpy)#
                    target_qpos_list.append(target_qpos)#
                    # rewards.append(ts.reward)
        #print("episode_id:", episode_id)#now use time as the identifier
        # visualize_differences(actiongt, actioninfer,plot_path=os.path.join(dataset_dir,f'episode_{episode_id}_qpos_{teornot}_verify.png'))
        #visualize_differences_more(qposgt, actiongt, actioninfer,
        #                           plot_path=os.path.join(dataset_dir,
        #                                                  f'episode_{episode_id}_qpos_{teornot}_verify.png'))
                    # plt.close()
                    #THEN IT IS THOSE TE THINGS
                #better also do interpolation to the joint states
                #if there is no enough points,
        '''
        motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, fill_value='extrapolate')#for loop里面来回的创建插值对象和函数吧

        joint_angle_interp = [
            interp1d(joint_time_buffer, [tup[i] for tup in joint_angle_buffer], fill_value='extrapolate') for i in
            range(len(joint_angle_buffer[0]))]

        def interpolate_joint_angles(time):
            return [interp_func(time) for interp_func in
                    joint_angle_interp]  # it is a list!#np.array([interp_func(time) for interp_func in joint_angle_interp])

        # joint_angle_interp = interp1d(joint_time_buffer, joint_angle_buffer, fill_value='extrapolate')
        joint_angular_velocity_interp = None  # interp1d(joint_time_buffer, joint_angular_velocity_buffer, fill_value='extrapolate')
        joint_effort_interp = None  # interp1d(joint_time_buffer, joint_effort_buffer, fill_value='extrapolate')

        actionpredicted=None
        #for idx, buffer_data in enumerate(image_body_buffer):#be accustom to the time of the camera
        for i in range(len(image_body_buffer)):#it is best to put it here!
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
                #then do the inference using pytorch
                #actionpredicted =pytorch model#
                #buffer_data.prediction=actionpredicted
                if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].joint_angle is not None:
                    episode_buffer[len(episode_buffer) - 1].action.append(motor_angle)  # now it has dimension 7
                    #lastgtaction=episode_buffer[len(episode_buffer) - 1].action
                    #lastprediction=episode_buffer[len(episode_buffer) - 1].prediction
                    #thisobservation=buffer_data.joint_angle
                    #error=thisobservation-lastprediction
                    #episode_buffer[len(episode_buffer) - 1].error = error#actionpredicted



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
        '''
    return #now you return something else, probably the array of errors#episode_buffer#so it is a list of Data objects

if __name__ == '__main__':
    rospy.init_node('verify_rosbag', anonymous=True)

    #if not os.path.exists(output_file):
    #    os.makedirs(output_file)
    #the main from imitate_episodes_xht goes here!
    #then you load the configuration and the policy
    #replay_buffer = ReplayBuffer.create_from_path(output_file, mode='a')
    maxlength=0
    episode_idx = 0#2
    for bag_file in os.listdir(bag_files_directory):
        if bag_file.endswith(".bag"):
            count=0
            #episode_data = process_bag_file(os.path.join(bag_files_directory, bag_file), bag_topics)
            episode_data = process_bag_file_realtime_infer(os.path.join(bag_files_directory, bag_file), bag_topics)
            buffer = []#Now I know that episode_data is a list of Data objects


            '''

            #the following data_dict is like the buffer in make_dataset
            data_dict = {  # it is a dictionary. This is important for data storage!
                '/observations/qpos': [],  # the value is a list
                '/observations/qvel': [],  # the value is a list
                '/action': [],#this action is the joint angles and the openness of the gripper
            }
            camera_names=['righthand','body']#,'lefthand']
            for cam_name in camera_names:  # top only in simulation#the value of this key is a list
                data_dict[f'/observations/images/{cam_name}'] = []  # in this case it will be /observations/images/top

            #for t in range(max_timesteps):  # 0-399
            for episode in episode_data[5:-5]:  # convert format skip first and last 5 stamps
                #if episode.tf is None:#So each episode is a Data object#because of -5,最后一个缺胳膊少腿没关系的！
                #    continue
                # print(np.shape(episode.img_camera))
                data = {#the data is a dictionary#this is actually the ts in record_sim_episodes
                    'camera_img': episode.img_camera,#align with the key that ACT uses
                    'camera_hand_img': episode.img_camera_hand,
                    #'tf': np.float32(episode.convert_tf_to_list(episode.tf)),
                    #'d_tf': np.float32(episode.delta_tf),
                    #'d_vel': np.float64(episode.vel),#you see it is already numpy#currently not needed
                    #'dist_gripper': np.float64(episode.motor_angle),#motor angle is the scalar measuring
                    'action': np.float64(episode.action),#np.hstack(episode.action, dtype=np.float32)
                    'qpos': np.float64(episode.joint_angle), #should be the position here! # needed
                    'qvel': np.float64(episode.joint_angular_velocity),  # currently not needed, also not provided
                    'effort': np.float64(episode.joint_effort)  # currently not needed, also not provided
                }

                #only use the pytorch model here!
                #current model

                buffer.append(data)#so buffer now is a list of dictionaries!

                data_dict['/observations/qpos'].append(data['qpos'])  #append(data['dist_gripper'])  # so it is a 2d list
                data_dict['/observations/qvel'].append(data['qvel'])  #.append(data['d_vel'])  # so it is a 2d list#how to get qvel from the real robot?
                data_dict['/action'].append(data['action'])  # so it is a list of 14d things
                #data_dict['/action'].append(data['action'])  # so it is a list of 14d things
                #for cam_name in camera_names:
                data_dict['/observations/images/righthand'].append(data['camera_hand_img'])
                data_dict['/observations/images/body'].append(data['camera_img'])
                #print(data['qpos'],data['action'])#it has both, since it is not yet the end of episode_data, where the last one lacks action
                #print('count',count)#,'data: ',data)
                count+=1

            print('count',count)
            maxlength=max(maxlength,count)#count is already the right data amount!
            # HDF5
            #t0 = time.time()

            #episode_idx=2#1#0
            max_timesteps=len(data_dict['/observations/qpos'])
            print('max_timesteps',max_timesteps,'maxlength',maxlength)#they should be the same now!
            #dataset_path = os.path.join(dataset_dir,'hdf5files', f'episode_{episode_idx}')
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:  # top, etc.
                    _ = image.create_dataset(cam_name, (max_timesteps, 240, 320, 3), dtype='uint8',
                                             chunks=(1, 240, 320, 3), )
                #qpos = obs.create_dataset('qpos', (max_timesteps, 14))  # are they really used or not?
                #qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                qpos = obs.create_dataset('qpos', (max_timesteps,7))  # are they really used or not?
                qvel = obs.create_dataset('qvel', (max_timesteps,7))
                action = root.create_dataset('action', (max_timesteps, 7))

                for name, array in data_dict.items():  # key value pairs
                    # print(name,len(array))#len(array) is always 400#array is a list of numpy arrays
                    #print(name, array[0].shape,type(array[0]))  #each element in array has dimension (7,) or (240,320,3) with type <class 'numpy.ndarray'>
                    root[name][...] = array
            #print(f'Saving: {time.time() - t0:.1f} secs\n')

            print(f'Saved to {dataset_dir}')
            episode_idx += 1
            #print(f'Success: {np.sum(success)} / {len(success)}')
            
            '''