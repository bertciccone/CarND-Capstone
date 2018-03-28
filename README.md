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
- DBW Node: Once the waypoint updater was publishing /final_waypoints, the waypoint_follower node started publishing messages to the/twist_cmd topic. After completing this step, the car drove in the simulator, ignoring the traffic lights.
- Detection/Classification: The detector detected the traffic light from the /image_color topic and a classifier determined its color. The topic /vehicle/traffic_lights contains the exact location and status of all traffic lights in simulator, so we used this ground truth to test our output.
- Waypoint publishing: Once we correctly identified the traffic light and determined its position, we converted it to a waypoint index and published it.
- Waypoint Updater (Full): We used /traffic_waypoint to change the waypoint target velocities before publishing to /final_waypoints. Our car now stops at red traffic lights and moves when they are green.

### 2) Design Features of Each Component

#### Waypoint Updater Node

#### DBW Node

#### Detection
Single shot detection (SSD) is used to find the traffic light in the camera image and create the detection box around the light. This cropping of the camera image consisting of just the 3 lights in a traffic light was passed to a classifier to determine whether it is red, yellow or red. After classification, the detector looks for at least 3 detections of the same light color before publishing the change to the /traffic_waypoint topic.

#### Classification

#### Waypoint publishing

#### Waypoint Updater


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
