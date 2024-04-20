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
import h5py
bag_files_directory ='/home/user/CODINGrepos/Imitation_Learning/record_data/' #sys.argv[1]
#output_file = sys.argv[1] + "/dataset"
dataset_dir=bag_files_directory#'/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag'
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
        self.action = None
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
    motor_angle_buffer = []
    episode_buffer = []
    motor_interp = None
    last_transform = None
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=task_time))
    image_body_buffer = []
    image_hand_buffer = []
    br = CvBridge()

    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=bag_topics):
            if topic == "/control/motor_angle":
                time = t.to_sec()
                angle = msg.data
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
        motor_interp = interp1d(motor_time_buffer, motor_angle_buffer, fill_value='extrapolate')

        for idx, buffer_data in enumerate(image_body_buffer):#be accustom to the time of the camera
            camera_time = buffer_data.time#buffer_data is a Data object
            '''
            # deal with tf
            if tf_buffer is not None:
                try:
                    joint_transform = tf_buffer.lookup_transform(target_frame=base_frame_id,
                                                                 source_frame=eff_frame_id,
                                                                 time=rospy.Time(camera_time))
                    buffer_data.tf = joint_transform
                    if len(episode_buffer) >= 1:
                        episode_buffer[len(episode_buffer) - 1].action = data.convert_tf_to_list(buffer_data.tf)
                    if last_transform is not None:
                        buffer_data.last_transform = last_transform
                        buffer_data.delta_tf_func()
                    last_transform = joint_transform
                except Exception as e:
                    # print(e)
                    buffer_data.tf = last_transform
            '''
            # deal with motor angle
            if motor_interp is not None:
                motor_angle = motor_interp(camera_time)#get the angle at the camera time!
                buffer_data.motor_angle = motor_angle
                if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].action is not None:
                    print("append motor angle!")#not entered when action is none
                    episode_buffer[len(episode_buffer) - 1].action.append(buffer_data.motor_angle.copy().tolist())
            # deal with vel
            vel = [0, 0, 0]
            if vel is not None:
                buffer_data.vel = vel
                if len(episode_buffer) >= 1 and episode_buffer[len(episode_buffer) - 1].action is not None:
                    print("append velocity!")
                    episode_buffer[len(episode_buffer) - 1].action.append(buffer_data.vel.copy())
            # deal with camera_hand
            if image_hand_buffer is not None:
                if idx < len(image_hand_buffer):
                    buffer_data.img_camera_hand = image_hand_buffer[idx]

            episode_buffer.append(buffer_data)

    return episode_buffer#so it is a list of Data objects


if __name__ == '__main__':
    rospy.init_node('read_rosbag', anonymous=True)

    #if not os.path.exists(output_file):
    #    os.makedirs(output_file)


    #replay_buffer = ReplayBuffer.create_from_path(output_file, mode='a')

    for bag_file in os.listdir(bag_files_directory):
        if bag_file.endswith(".bag"):
            count=0
            episode_data = process_bag_file(os.path.join(bag_files_directory, bag_file), bag_topics)
            buffer = []#Now I know that episode_data is a list of Data objects




            #the following data_dict is like the buffer in make_dataset
            data_dict = {  # it is a dictionary. This is important for data storage!
                '/observations/qpos': [],  # the value is a list
                '/observations/qvel': [],  # the value is a list
                #'/action': [],
            }
            camera_names=['righthand','body']#,'lefthand']
            for cam_name in camera_names:  # top only in simulation#the value of this key is a list
                data_dict[f'/observations/images/{cam_name}'] = []  # in this case it will be /observations/images/top






            for episode in episode_data[5:-5]:  # convert format skip first and last 5 stamps
                #if episode.tf is None:#So each episode is a Data object
                #    continue
                # print(np.shape(episode.img_camera))
                data = {#the data is a dictionary#this is actually the ts in record_sim_episodes
                    'camera_img': episode.img_camera,#align with the key that ACT uses
                    'camera_hand_img': episode.img_camera_hand,
                    #'tf': np.float32(episode.convert_tf_to_list(episode.tf)),
                    #'d_tf': np.float32(episode.delta_tf),
                    'd_vel': np.float32(episode.vel),
                    'dist_gripper': np.float32(episode.motor_angle),
                    #'action': np.hstack(episode.action, dtype=np.float32)
                }



                buffer.append(data)#so buffer now is a list of dictionaries!
                '''
                data_dict['/observations/qpos'].append(ts.observation['qpos'])  # so it is a 2d list
                data_dict['/observations/qvel'].append(ts.observation['qvel'])  # so it is a 2d list
                data_dict['/action'].append(action)  # so it is a list of 14d things
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                '''

                data_dict['/observations/qpos'].append(data['dist_gripper'])  # so it is a 2d list
                data_dict['/observations/qvel'].append(data['d_vel'])  # so it is a 2d list#how to get qvel from the real robot?
                #data_dict['/action'].append(data['action'])  # so it is a list of 14d things
                #for cam_name in camera_names:
                data_dict['/observations/images/righthand'].append(data['camera_hand_img'])
                data_dict['/observations/images/body'].append(data['camera_img'])

                print('count',count)#,'data: ',data)
                count+=1

            '''
            data_dict = dict()
            #print("len(buffer)",len(buffer),"len(episode_data)",len(episode_data))#587,587
            for key in buffer[0].keys():#buffer[0] is a dictionary, thus its key is also the key in other dictionaries in the list
                data_list = []
                for x in buffer:#buffer now is a list of dictionaries!
                    data_list.append(x[key])#data_list is just a list of things
                data_dict[key] = np.stack(data_list)
                print('key',key,'data_dict[key].shape()',data_dict[key].shape)
            #replay_buffer.add_episode(data_dict, compressors='disk')

            #key camera_img data_dict[key].shape()(587, 240, 320, 3)
            #key camera_hand_img data_dict[key].shape()(587, 240, 320, 3)
            #key d_vel data_dict[key].shape()(587, 3)
            #key dist_gripper data_dict[key].shape()(587, )
            '''

            '''
            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            joint_traj = joint_traj[:-1]  # still within an episode/one trajectory
            episode_replay = episode_replay[:-1]
            # print("len(joint_traj)",len(joint_traj))#400#
            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(joint_traj)  # 400 now
            while joint_traj:  # it is the action
                action = joint_traj.pop(0)  # action is a 14 dimensional list
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])  # so it is a 2d list
                data_dict['/observations/qvel'].append(ts.observation['qvel'])  # so it is a 2d list
                data_dict['/action'].append(action)  # so it is a list of 14d things
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                # print("action,qpos,qvel,image[top]:",action.shape,ts.observation['qpos'].shape,ts.observation['qvel'].shape,ts.observation['images']['top'].shape)
                # action, qpos, qvel, image[top]: (14,)(14, )(14, )(480, 640, 3)
            '''

            # HDF5
            #t0 = time.time()

            episode_idx=1#0
            max_timesteps=len(data_dict['/observations/qpos'])
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:  # top, etc.
                    _ = image.create_dataset(cam_name, (max_timesteps, 240, 320, 3), dtype='uint8',
                                             chunks=(1, 240, 320, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                #qpos = obs.create_dataset('qpos', (max_timesteps, 14))  # are they really used or not?
                #qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                qpos = obs.create_dataset('qpos', (max_timesteps,))  # are they really used or not?
                qvel = obs.create_dataset('qvel', (max_timesteps,3))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():  # key value pairs
                    # print(name,len(array))#len(array) is always 400#array is a list of numpy arrays
                    print(name, array[0].shape,type(array[0]))  #each element in array has dimension (14,) or (480,640,3)
                    root[name][...] = array
            #print(f'Saving: {time.time() - t0:.1f} secs\n')

            print(f'Saved to {dataset_dir}')
            #print(f'Success: {np.sum(success)} / {len(success)}')