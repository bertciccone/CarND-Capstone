# Self Driving Car Nanodegree Capstone Project

## Team: Carla's Quest
### Members:
- Bert Ciccone
- Johan Smet
- Josef Steinbaeck
- Albert Vo
- Tobias Wagner

The goals of this project are to:
- Create a map waypoint updater, a twist controller and a traffic light detector/classifier in Python and ROS
- To enable this software to drive a car in a Unity simulator in a lane on the highway, driving under the maximum speed limit and stopping at all traffic lights
- To work as a team comprised of five members, each contributing development to the project and then integrating and testing the resulting system
- Test the software using rosbag test data captured from a previous run of the software in a real self-driving car
- To run the software in Udacity's test track
- Finally, to review the captured rosbag and analyze it in ROS rviz

### 1) Order of Development

The team followed this order of development to build and test the components until the system was completed as a whole:
- Waypoint Updater Node (Partial): We completed a partial waypoint updater which subscribes to /base_waypoints and /current_pose and publishes to /final_waypoints.
- Twist Controller Node: Once the waypoint updater was publishing /final_waypoints, the waypoint_follower node started publishing messages to the/twist_cmd topic. After completing this step, the car drove in the simulator, ignoring the traffic lights.
- Detection/Classification: The detector detected the traffic light from the /image_color topic and a classifier determined its color. The topic /vehicle/traffic_lights contains the exact location and status of all traffic lights in simulator, so we used this ground truth to test our output.
- Waypoint publishing: Once we correctly identified the traffic light and determined its position, we converted it to a waypoint index and published it.
- Waypoint Updater (Full): We used /traffic_waypoint to change the waypoint target velocities before publishing to /final_waypoints. Our car now stops at red traffic lights and moves when they are green.

### 2) Design Features of Each Component

#### Waypoint Updater

The waypoint updater node takes the list of waypoints and the current position of the car to find the next 200 waypoints using a kd-tree lookup. These waypoints are then published every 0.1 second to the topic /final_waypoints.

The waypoint updater node subscribes to the traffic_waypoint topic. In case of an upcoming red traffic light, the speed of the following waypoints is adjusted in order to stop right before the traffic light. The current speed of the car is considered in order to determine the braking distance and set the waypoint speed accordingly. If a too high de-acceleration would be required to brake before the traffic light, the vehicle will skip the light.

#### Twist Controller

The twist-controller node uses the output of the waypoint-updater node to send control commands to the car. A PID-controller handles throttling and braking, while a low-pass filter smooths the steering commands. The gains of the PID-controller can be tuned online using the ROS dynamic_reconfigure package.

#### Traffic Light Detection/Classification

The traffic light detector uses a pre-trained SSD Mobilenet model to do object detection.  When the detector finds a traffic light, the detector feeds just the image of the traffic light to the classifier.  The classifier is a CNN that consists of the following layers:

Layer (type)                 Output Shape              Param #  

conv2d_1 (Conv2D)            (None, 32, 32, 32)        896      
activation_1 (Activation)    (None, 32, 32, 32)        0        
conv2d_2 (Conv2D)            (None, 30, 30, 32)        9248      
activation_2 (Activation)    (None, 30, 30, 32)        0        
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0        
dropout_1 (Dropout)          (None, 15, 15, 32)        0        
conv2d_3 (Conv2D)            (None, 15, 15, 64)        18496    
activation_3 (Activation)    (None, 15, 15, 64)        0        
conv2d_4 (Conv2D)            (None, 13, 13, 64)        36928    
activation_4 (Activation)    (None, 13, 13, 64)        0        
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0        
dropout_2 (Dropout)          (None, 6, 6, 64)          0        
flatten_1 (Flatten)          (None, 2304)              0        
dense_1 (Dense)              (None, 512)               1180160  
activation_5 (Activation)    (None, 512)               0        
dropout_3 (Dropout)          (None, 512)               0        
dense_2 (Dense)              (None, 3)                 1539      
activation_6 (Activation)    (None, 3)                 0        

Total params: 1,247,267
Trainable params: 1,247,267
Non-trainable params: 0

We trained the classifier model using Red, Yellow and Green traffic light images captured from the simulator's vehicle camera.
The classifier loads the trained model weights, subscribes to the camera topic and performs inference and classification of traffic lights based on the raw images.

Please use **one** of the two installation options, either native **or** docker installation.

### 3) Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### 4) Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### 5) Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### 6) Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### 7) Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
