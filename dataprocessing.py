'''
import bagpy
from bagpy import bagreader

b = bagreader('/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag/2024-04-07-18-07-15.bag')#('09-23-59.bag')

# get the list of topics
print(b.topic_table)
#print(b['joint_states'])

# get all the messages of type velocity
velmsgs   = b.vel_data()
veldf = pd.read_csv(velmsgs[0])
plt.plot(veldf['Time'], veldf['linear.x'])

# quickly plot velocities
b.plot_vel(save_fig=True)

# you can animate a timeseries data
bagpy.animate_timeseries(veldf['Time'], veldf['linear.x'], title='Velocity Timeseries Plot')
'''

import rosbag

# Path to your ROS bag file
bag_path = '/home/user/bagfilesxht2/pick_and_place_rosbag/0407_rosbag/2024-04-07-18-07-15.bag'#"your_bag_file.bag"

# Open the bag file
bag = rosbag.Bag(bag_path)

# Iterate through messages in the bag file
for topic, msg, t in bag.read_messages():
    # Process messages here
    #print(f"Topic: {topic}, Message: {msg}, Timestamp: {t}")
    print(f"Topic: {topic}, Timestamp: {t}")
# Close the bag file
bag.close()