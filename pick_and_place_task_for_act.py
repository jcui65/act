import time
import numpy as np
import torch
#import dill
#import hydra
import scipy.spatial.transform as st
#from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
#from diffusion_policy.common.pytorch_util import dict_apply
#from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os
# ros relate
import rospy
from cv_bridge import CvBridge
from collections import deque
import transformations as tflib
import tf
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64, Header
from geometry_msgs.msg import PoseArray, Pose


#ACT relate
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
from constants_xht import DT#I make modification from constants.py in the repo of ACT
from constants_xht import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy#see the repo of ACT
from constants_xht import NUMBEROFARMS,DOFLEFT,DOFRIGHT

DEBUG = False
'''
#not needed if using ACT
class DiffusionPolicy:#For me it should be the ACT policy
    def __init__(self, ckpt_path):
        # load checkpoint and get configuration
        payload = torch.load(ckpt_path, pickle_module=dill)
        self.cfg = payload["cfg"]
        cls = hydra.utils.get_class(self.cfg._target_)#it is a configuration thing!
        self.n_obs_steps = self.cfg.n_obs_steps
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        # configure model
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model
        self.device = torch.device("cuda")
        self.policy.eval().to(self.device)
        self.policy.num_inference_steps = 16  # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

    def infer(self, obs):  # need warming up; the first infer will cost about 1.2s, the latter will cost about 0.1s
        with torch.no_grad():
            self.policy.reset()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs, shape_meta=self.cfg.task.shape_meta
            )
            obs_dict = dict_apply(#trochification function
                obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device),
            )
            result = self.policy.predict_action(obs_dict)#throw into the network for inference
            action = result["action"][0].detach().to("cpu").numpy()
        return action
'''

class RosNodeACT:
    def __init__(self, policy):
        self.action = None
        self.policy = policy
        self.eff_frame_id = "eef"#since no tf is needed, I don't need to care about it
        self.base_frame_id = "body"#since no tf is needed, I don't need to care about it
        self.br = CvBridge()
        self.image_w = 320
        self.image_h = 240
        dt = DT#1/30#1 / 10  # 10HZ
        self.n_obs_steps = 1#policy.n_obs_steps#I think for me this is 1
        self._init_buffer(self.n_obs_steps, 5)#line 89
        self.camera_body_sub = rospy.Subscriber(
            "/camera/color/image_raw/compressed",
            CompressedImage,
            self.camera_body_callback,
        )
        self.camera_hand_sub = rospy.Subscriber(
            "/camera_hand/color/image_raw/compressed",
            CompressedImage,
            self.camera_hand_callback,
        )#I also need to subscribe to joint angles
        self.current_joint_states_sub=rospy.Subscriber("/joint_states",JointState,self.joint_states_callback)
        if DEBUG:
            self.gripper_pub = rospy.Publisher("/infer/motor_angle", Float64, queue_size=5)
        else:
            self.gripper_pub = rospy.Publisher("/control/motor_angle", Float64, queue_size=5)
        self.pose_pub = rospy.Publisher("/infer/robot_pose", PoseArray, queue_size=5)#pose_pub is not used in ACT! I need to publish joint angles!
        # self.pose_pub = rospy.Publisher("/infer/robot_pose", Pose, queue_size=5)#not needed in ACT
        self.qpos_pub=rospy.Publisher("/joint_states",JointState,queue_size=5)
        self.timer = rospy.Timer(rospy.Duration(dt), self.timer_callback)
        self.tt=0#I want to use tt time count to count how many times the timer_callback is called!
        self.tf_listener = tf.TransformListener()

    def _init_buffer(self, n_obs_step: int, expand_step: int):#line 69
        deque_max_len = n_obs_step + expand_step#is this expand_step a kind of redundancy?
        self.camera_body_buffer = deque(maxlen=deque_max_len)
        self.camera_hand_buffer = deque(maxlen=deque_max_len)
        self.gripper_buffer = deque([100.0, 100.0], maxlen=deque_max_len)
        self.last_n_delta_tf = None#I don't need this!
        self.joint_states_buffer=deque(maxlen=deque_max_len)#I need this to get previous joint states

    def camera_body_callback(self, msg):
        # print("camera_body_callbacks")
        camera_time = msg.header.stamp.to_sec()
        img_camera = self.br.compressed_imgmsg_to_cv2(msg)
        image_body_dict = {"camera_time": camera_time, "img": img_camera}
        self.camera_body_buffer.append(image_body_dict)

    def camera_hand_callback(self, msg):
        # print("camera_hand_callback")
        camera_hand_time = msg.header.stamp.to_sec()
        img_camera_hand = self.br.compressed_imgmsg_to_cv2(msg)
        image_hand_dict = {"camera_time": camera_hand_time, "img": img_camera_hand}
        self.camera_hand_buffer.append(image_hand_dict)

    def joint_states_callback(self, msg):#well, actually the joint_states include qpos, qvel and qeffort.
        # print("camera_hand_callback")#In ACT, only qpos is used
        joint_states_time = msg.header.stamp.to_sec()
        joint_states = msg#.position#self.br.compressed_imgmsg_to_cv2(msg)
        joint_states_dict = {"js_time": joint_states_time, "js": joint_states}
        self.joint_states_buffer.append(joint_states_dict)

    def timer_callback(self, event):
        before_time = time.time()
        self.tt+=1
        if len(self.camera_body_buffer) > 2:
            last_n_camera_body = list(self.camera_body_buffer)[-1]#[
            #    -self.n_obs_steps - 1 :#for me this -1 is not necessary
            #]  # -1 is for delta_tf compute#In ACT, n=1 is enough
            last_n_camera_hand = list(self.camera_hand_buffer)[-1]#[-self.n_obs_steps :]#for me, it is just the last one!
            last_n_joint_states=list(self.joint_states_buffer)[-1]#[-self.n_obs_steps :]
            last_n_time = last_n_camera_body["camera_time"]#[x["camera_time"] for x in last_n_camera_body]#for me, n=1!!!#get the time!
            last_n_camera_body_img = last_n_camera_body["img"]#[x["img"] for x in last_n_camera_body]#get the image
            last_n_camera_hand_img = last_n_camera_hand["img"]#[x["img"] for x in last_n_camera_hand]
            last_n_qpos = last_n_joint_states["js"].position#[x["qpos"] for x in last_n_joint_states]
            last_n_gripper = list(self.gripper_buffer)[-1]#[-self.n_obs_steps :]
            #last_n_tf = list()
            # print("last_n_time", last_n_time)
            last_n_qpos7 =last_n_qpos+last_n_gripper#concatenating 2 lists as 7=6+1
            '''
            for ros_time in last_n_time:
                try:
                    trans, rot = self.tf_listener.lookupTransform(
                        target_frame=self.base_frame_id,
                        source_frame=self.eff_frame_id,
                        time=rospy.Time(ros_time),
                    )
                    last_n_tf.append(np.hstack([trans, rot]))
                except Exception as e:
                    print(e)

            if len(last_n_tf) > self.n_obs_steps:
                self.last_n_delta_tf = self.delta_tf_func(last_n_tf)

            last_n_vel = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]#just a place holder?
            last_n_tf[:] = last_n_tf[: self.n_obs_steps]
            '''
            if True:#self.last_n_delta_tf is not None and len(last_n_gripper) > 0:#seems not used in ACT!
                '''
                obs = {
                    "camera_img": np.array(last_n_camera_body_img),
                    "camera_hand_img": np.array(last_n_camera_hand_img),
                    #"tf": np.array(last_n_tf),
                    #"d_tf": np.array(self.last_n_delta_tf),
                    #"d_vel": np.array(last_n_vel),
                    "qpos": np.array(last_n_qpos),
                    "dist_gripper": np.array(last_n_gripper),#[-2:]),
                }
                '''
                image_dict = {#form the image_dict which will be processed later to get the image
                    "camera_img": np.array(last_n_camera_body_img),
                    "camera_hand_img": np.array(last_n_camera_hand_img),
                    # "tf": np.array(last_n_tf),
                    # "d_tf": np.array(self.last_n_delta_tf),
                    # "d_vel": np.array(last_n_vel),
                    #"qpos": np.array(last_n_qpos),
                    #"dist_gripper": np.array(last_n_gripper),  # [-2:]),
                }
                camera_names=list(image_dict.keys())
                with torch.inference_mode():#顾名思义，inference
                    qpos_numpy=np.asarray(last_n_qpos7)#from list to array
                    qpos = pre_process(
                        qpos_numpy)  # normalization#qpos_numpy#because I didn't make the right preprocessing of the data!#
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    # qpos_history[:, t] = qpos#I need to check the camera_names list to see what it contains
                    # curr_image = get_image(ts, camera_names)#camera_names contains all the cameras, get all images at ts
                    # I need to build the image_dict
                    curr_image = get_image_xht(image_dict, camera_names)  # , t)  #我是先装手还是先装body来着？
                    query_frequency = policy_config['num_queries']
                    #action = self.policy.infer(obs)#so obs is a dictionary of lists
                    #action=self.policy(qpos, image)
                    #self.tt means the time counts, each time the timer_callback is called, tt+=1. Is my way of counting the number of calls to timer_callback correct?
                    if self.tt % query_frequency == 0:  # the action only feeds in every query_frequency steps?
                        all_actions = policy(qpos, curr_image)  # what does all_actions contain?
                    if temporal_agg:  # all_actions should have 100 in one of its dimension
                        # print("num_queries",num_queries)#100 is correct!
                        all_time_actions[self.tt % num_queries] = all_actions  # that line is overwritten!
                        # all_time_actions[[tt], tt:tt + num_queries] = all_actions  # 100 things, right?
                        # all_time_actions[tt, tt:tt + num_queries] = all_actions  # 100 things, right?
                        # print("all_time_actions2.shape:", all_time_actions.shape)  # (500,600,14)#torch.Size([1228, 1328, 7]) also correct!#[[t], t:t])
                        # print("all_time_actions:",len(all_time_actions),len(all_time_actions[0]),len(all_time_actions[0][0]))#400 500 14
                        if (self.tt >= num_queries - 1):#this is different from the original implementation of ACT and thus saves much space
                            rowindex = torch.arange(num_queries)
                            columnindex = (torch.arange(self.tt, self.tt - 100, -1)) % num_queries
                        else:
                            rowindex = torch.arange(self.tt + 1)
                            columnindex = torch.arange(self.tt, -1, -1)
                        # actions_for_curr_step = all_time_actions[:, tt]  # num_queries is 100
                        actions_for_curr_step = all_time_actions[rowindex, columnindex]  # num_queries is 100
                        # print("actions_for_curr_step:", actions_for_curr_step)  #
                        # print("actions_for_curr_step.shape:",actions_for_curr_step.shape)#(500,14)#torch.Size([1228, 7]) also correct!#print("size0,size1:",len(actions_for_curr_step),len(actions_for_curr_step[0]))#400,14#1228,7
                        actions_populated = torch.all(actions_for_curr_step != 0,
                                                    axis=1)  # it is a mask#operating on the dimension that is 14 or 7
                        # print("actions_populated",actions_populated)
                        # if(len(actions_populated)>0):#len(actions_populated)=400 always
                        # print("actions_populated.shape:",actions_populated.shape)#(500)#torch.Size([1228]) also correct!#print("len(actions_populated),len(actions_populated[1]):",len(actions_populated),len(actions_populated[0]))#1228
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # if(len(actions_for_curr_step)>0):#actions_for_curr_step.shape=torch.Size([1-100, 14])#1-100 means from 1 to 100
                        # print(f"t={tt},actions_for_curr_step.shape:",actions_for_curr_step.shape)#(1,14)-(100,14)#torch.Size([1228, 7]), correct!#print("tt,len(afcs),len(afcs[1]):",tt, len(actions_for_curr_step),actions_for_curr_step.shape)#,len(actions_for_curr_step[0]))
                        # t,len(afcs),len(afcs[1]): 99 100 torch.Size([100, 14])#t,len(afcs),len(afcs[1]): 100 100 torch.Size([100, 14])
                        # print("len(actions_for_curr_step)",len(actions_for_curr_step))
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        # print("exp_weights.shape:", exp_weights.shape)  # (1,) to (100,)#(1,) to (100,)#correct!
                        exp_weights = exp_weights / exp_weights.sum()  # normalize? Yeah, I think so
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # print("exp_weights.shape2:", exp_weights.shape)  # (1,1) to (100,1)#(1,1) to (100,1), correct! because of the unsqueeze#
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0,
                                                                            keepdim=True)  # kind of like convolution
                        # print("raw_action.shape", raw_action.shape)  # (1,14)#torch.Size([1, 7]) correct!
                    else:
                        raw_action = all_actions[:,
                                    self.tt % query_frequency]  # really one piece after another! 100-100-100-100
                    action=raw_action
                    self.action = action#raw_action#
                '''
                robot_pose_array = PoseArray()
                robot_pose_array.header.frame_id = "body"
                robot_pose_array.header.stamp = rospy.Time.now()
                # # only publish the first
                # robot_pose = Pose()
                # (
                #     robot_pose.position.x,
                #     robot_pose.position.y,
                #     robot_pose.position.z,
                # ) = action[0][:3]
                # (
                #     robot_pose.orientation.x,
                #     robot_pose.orientation.y,
                #     robot_pose.orientation.z,
                #     robot_pose.orientation.w,
                # ) = action[0][3:7]
                # robot_pose_array.poses.append(robot_pose)
                
                # publish the sequence
                if DEBUG:
                    for i in range(len(action)):
                        robot_pose = Pose()
                        (
                            robot_pose.position.x,
                            robot_pose.position.y,
                            robot_pose.position.z,
                        ) = action[i][:3]
                        (
                            robot_pose.orientation.x,
                            robot_pose.orientation.y,
                            robot_pose.orientation.z,
                            robot_pose.orientation.w,
                        ) = action[i][3:7]
                        robot_pose_array.poses.append(robot_pose)
                else:#if not DEBUG
                    
                    #It is not needed in ACT
                    for i in range(10):
                        robot_pose = Pose()
                        (
                            robot_pose.position.x,
                            robot_pose.position.y,
                            robot_pose.position.z,
                        ) = action[i][:3]
                        (
                            robot_pose.orientation.x,
                            robot_pose.orientation.y,
                            robot_pose.orientation.z,
                            robot_pose.orientation.w,
                        ) = action[i][3:7]
                        robot_pose_array.poses.append(robot_pose)
                '''
                gripper_data = action[6]#action[-1][7]
                gripper_dist = Float64()
                gripper_dist.data = gripper_data

                # print("pub msg", robot_pose_array)
                self.gripper_pub.publish(gripper_dist)
                #self.pose_pub.publish(robot_pose_array)
                # print("finish pub")
                self.gripper_buffer.append(gripper_data)#here, you not use both gripper as input and output, so you need to append to the buffer in both cases, right?

                qpos_data=action[:6]#first 6 components
                expected_joint_states=JointState()
                expected_joint_states.header = Header()
                expected_joint_states.header.stamp = rospy.Time.now()
                expected_joint_states.position=qpos_data#6 dimensional
                expected_joint_states.name=[]#["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]#
                #the above names are got from ALOHA. Kuihan please check and correct if the names of joints are different on our robot
                expected_joint_states.velocity=[]
                expected_joint_states.effort=[]
                self.qpos_pub.publish(expected_joint_states)
                # print("infer_duration:", time.time() - before_time)
                # print("action", action)

    def delta_tf_func(self, tf_arr):
        list_delta_tf = []
        for idx in range(len(tf_arr) - 1):
            tf_list_current = tf_arr[idx]
            tf_list_future = tf_arr[idx + 1]
            current_translation, current_rotation = (
                tf_list_current[:3],
                tf_list_current[-4:],
            )
            future_translation, future_rotation = (
                tf_list_future[:3],
                tf_list_future[-4:],
            )
            c_rotation = tflib.quaternion_matrix(current_rotation)#current rotation
            c_traslation = tflib.translation_matrix(current_translation)
            c_transformation_matrix = c_rotation @ c_traslation
            f_rotation = tflib.quaternion_matrix(future_rotation)#future rotation
            f_translation = tflib.translation_matrix(future_translation)
            f_translation_matrix = f_rotation @ f_translation
            delta_transform = (
                tflib.inverse_matrix(c_transformation_matrix) @ f_translation_matrix
            )
            delta_translation = tflib.translation_from_matrix(delta_transform)
            delta_rotation = tflib.quaternion_from_matrix(delta_transform)
            list_delta_tf.append(np.hstack([delta_translation, delta_rotation]))
        return list_delta_tf

    def tf2list(self, tf):
        list_tf = [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        return list_tf

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)#see policy.py!
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def get_image_xht(image_dict, camera_names):#, t):#where to put this function is a question
    curr_images = []#put this function outside the function may save a little time?
    for cam_name in camera_names:  # ['top'] only? Yes in simulation
        curr_image = rearrange(image_dict[cam_name],#[t],
                               'h w c -> c h w')  # don't quite know the details, but know what it means!
        curr_images.append(curr_image)  # so current_images contains images in all angles/aspects/viewpoints
    curr_image = np.stack(curr_images, axis=0)  # stack current images together
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # normalize and other processing
    return curr_image

if __name__ == "__main__":
    rospy.init_node("policy_infer_node_act")#this is the node name for inference!

    rospy.loginfo("staring policy inference...")

    #policy = DiffusionPolicy(
    #    "/home/user/work/checkpoints_trained/0409_right_pick_and_place.ckpt"
    #)#So it is an instantiation of the policy class!
    parser = argparse.ArgumentParser()
    #parser.add_argument('--eval', action='store_true')#usually eval, right?#Since it is on real robot, then it will be eval, rather than train
    #parser.add_argument('--onscreen_render', action='store_true')#In simulation, I need to render it. But in real robots, I think I don't need this!
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)#need to get checkpoint directory
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True,default='ACT')#normally ACT, but maybe can also try CNNMLP
    #parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)#this's important, and maybe can be incorporated with Shenhong's embedding!
    #parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)#if just for eval, then no need to use this!
    #parser.add_argument('--seed', action='store', type=int, help='seed', required=True)#why need this during inference?
    #parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)#only for training
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)#training only

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)#training only
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)#also test
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)#training only
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)#training only
    parser.add_argument('--temporal_agg', action='store_true')#also test

    args=vars(parser.parse_args())
    #args = parser.parse_args()



    #rospy.init_node('verify_rosbag', anonymous=True)

    #if not os.path.exists(output_file):
    #    os.makedirs(output_file)
    #the main from imitate_episodes_xht goes here!
    #then you load the configuration and the policy
    #replay_buffer = ReplayBuffer.create_from_path(output_file, mode='a')

    #set_seed(1)#so this is not critical anymore
    # command line parameters
    #is_eval = args['eval']#eval or train
    ckpt_dir = args['ckpt_dir']#
    policy_class = args['policy_class']#ACT or other baseline methods!
    #onscreen_render = args['onscreen_render']
    #task_name = args['task_name']#cube transfer, insertion, etc.
    #batch_size_train = args['batch_size']#8 default
    #batch_size_val = args['batch_size']#the val batchsize is the same as the train batch size
    #num_epochs = args['num_epochs']#2000 default

    # get task parameters
    '''
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        #from constants import SIM_TASK_CONFIGS
        from constants_xht import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    '''
    #from constants_xht import SIM_TASK_CONFIGS
    #task_config = SIM_TASK_CONFIGS[task_name]
    #dataset_dir = task_config['dataset_dir']#I just don't need this!
    #num_episodes = task_config['num_episodes']#50 default
    #episode_len = task_config['episode_len']#400 default
    #camera_names = task_config['camera_names']#it should be a list of camera_names#it seems that in simulation, it only contain one viewpoint, which is from the top!

    # fixed parameters
    if NUMBEROFARMS==2:
        state_dim=DOFRIGHT+DOFLEFT#DOF of the right arm
    elif NUMBEROFARMS==1:
        state_dim=DOFRIGHT#DOF of the right arm
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
                         'hidden_dim': args['hidden_dim'],#training only
                         'dim_feedforward': args['dim_feedforward'],#3200 by default
                         'lr_backbone': lr_backbone,#lr_backbone means learning rate for the backbone
                         'backbone': backbone,
                         'enc_layers': enc_layers,#that is not modifyable
                         'dec_layers': dec_layers,#not modifyable
                         'nheads': nheads,#not modifyable
                         #'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':#
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,}#there is no action chunking in CNNMLP
                         #'camera_names': camera_names,}#training only
    else:
        raise NotImplementedError
    '''''' 
    config = {
        #'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        #'episode_len': episode_len,
        'state_dim': state_dim,#14 as shown in the paper
        'lr': args['lr'],
        'policy_class': policy_class,
        #'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        #'task_name': task_name,
        #'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],#is it a val thing or is it also a train thing?
        #'camera_names': camera_names,
        #'real_robot': not is_sim
    }




    #set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    #real_robot = config['real_robot']#not is_sim
    policy_class = config['policy_class']#ACT or something else
    #onscreen_render = config['onscreen_render']#currently useless
    policy_config = config['policy_config']
    #camera_names = config['camera_names']
    #max_timesteps = config['episode_len']#400
    #task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    #onscreen_cam = 'angle'

    # load policy and stats
    ckpt_name = f'policy_best.ckpt'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)#121
    #policy = make_policy()
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

    query_frequency = policy_config['num_queries']  # 1 for CNNMLP, 100 for ACT, if not temporal_agg then open loop 100, otherwise temporal_agg
    if temporal_agg:  # since this is better, I will choose this?
        query_frequency = 1  # temporal agg only matters in evaluation!
        num_queries = policy_config['num_queries']
        all_time_actions = torch.zeros([num_queries, num_queries, state_dim]).cuda()  # just once!
        k = 0.01
    ##tt = 0
    rosnode = RosNodeACT(policy)

    rospy.spin()
