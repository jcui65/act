import pathlib
###abstraction barrier
NUMBEROFARMS=1#2#
DOFLEFT=7
DOFRIGHT=7
IMAGEWIDTH=320#seems not important, as the network will know how to determine the size of the images?
IMAGEHEIGHT=240
### Task parameters
DATA_DIR = '/home/user/bagfilesxht/pick_and_place_rosbag/0413_rosbag/green'#/ff2'#'selfcollecteddata'#'/home/user/bagfilesxht2/pick_and_place_rosbag/0408_rosbag/left'#'selfcollecteddata'#'/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag'##'selfcollecteddata/sim_insertion_scripted'#jianning cui makes this change#'<put your data dir here>'
SIM_TASK_CONFIGS = {#it is a dictionary with task name as the key and its configurations as the value
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,#there is a difference
        'camera_names': ['top']
    },
    'sim_pick_and_place_left_human': {#'sim_pick_and_place_human': {
        'dataset_dir': DATA_DIR + '/pick_and_place_left_human_4',#'/pick_and_place_left_human_8',#'/pick_and_place_left_human_2',#'/pick_and_place_left_human',#'/pick_and_place_human2',#'/pick_and_place_human',#'hdf5files',#'/sim_insertion_human',
        'num_episodes': 50,#51,#the above 2 means the time difference is 2!
        'episode_len': 100,#1228,#500,#there is a difference
        'camera_names': ['body','righthand']#I set them#['top']
    },
    'sim_pick_and_place_right_human': {#'sim_pick_and_place_human': {
        'dataset_dir': DATA_DIR + '/pick_and_place_right_human',#'/pick_and_place_human2',#'/pick_and_place_human',#'hdf5files',#'/sim_insertion_human',
        'num_episodes': 50,#51,#
        'episode_len': 100,#951,#500,#there is a difference
        'camera_names': ['body','righthand']#I set them#['top']
    },
    'sim_pick_and_place_green_human': {#'sim_pick_and_place_human': {
        'dataset_dir': DATA_DIR + '/pick_and_place_green_human_2',#'/pick_and_place_green_human',#'/pick_and_place_human2',#'/pick_and_place_human',#'hdf5files',#'/sim_insertion_human',
        'num_episodes': 99,#6,#50,#51,#
        'episode_len': 100,#1228,#500,#there is a difference
        'camera_names': ['body','righthand']#I set them#['top']
    },
}

### Simulation envs fixed constants
DT = 1/30#0.02#yeah, 50 hz, corresponds well with the paper!
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]#[0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
#why START_ARM_POSE has 16 dimensions? because 0.2239 is for the left claw, while -0.2239 is for the right claw!
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910#it is bigger than 1, so certainly it is not normalized
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)#between 0 and 1
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)#from unnormalized to normalized
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE#between close and open
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))#position, end effector, decartes

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)#joint space
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))
#no jacobian in this process?
MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2#medium position
