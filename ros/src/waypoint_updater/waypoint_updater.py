#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math
import numpy as np

# The k-d tree library from https://github.com/stefankoegl/kdtree
import kdtree

# Import the Python profiling package for performance timing
import cProfile

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

DEBUG_LEVEL = 1  # 0 no Messages, 1 Important Stuff, 2 Everything

UPDATE_FREQUENCY = 10  # 10Hz should do the trick :)

# This is a k-d tree item containing containing waypoint x, y coordinates as keys and waypoint indexes as data
class Item(object):
    def __init__(self, x, y, data):
        self.coords = (x, y)
        self.data = data

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.coords[0], self.coords[1], self.data)

class WaypointUpdater(object):
    def __init__(self):

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.current_pose = None
        self.next_wp_index = None

        # For k-d tree test
        self.waypoints_kdtree = None

        if DEBUG_LEVEL >= 1: rospy.logwarn("Waypoint Updater loaded!")
        rate = rospy.Rate(UPDATE_FREQUENCY)

        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()
            # rospy.spin()

    def loop(self):
        if (self.current_pose is not None) and (self.base_waypoints is not None):

            #cProfile.runctx('self.get_next_waypoint_index()', globals(), locals(), 'get_next_waypoint_index.log')
            #next_wp_index = self.get_next_waypoint_index()

            #cProfile.runctx('self.get_next_waypoint_index2()', globals(), locals(), 'get_next_waypoint_index2.log')
            next_wp_index = self.get_next_waypoint_index2()

            # To display profile log file in command line window:
            # python
            # >>> import pstats
            # >>> pstats.Stats('./src/waypoint_updater/get_next_waypoint_index.log').print_stats()
            # >>> pstats.Stats('./src/waypoint_updater/get_next_waypoint_index2.log').print_stats()

            self.publish_waypoints(next_wp_index)

    def pose_cb(self, msg):
        # When we get a new position then load it to the local variable
        self.current_pose = msg.pose

    def waypoints_cb(self, waypoints):
        # When we the waypoints we save them to the local variable
        self.base_waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    # Helper functions
    def get_next_waypoint_index(self):
        # prepare car position and orientation
        car_x = self.current_pose.position.x
        car_y = self.current_pose.position.y
        s = self.current_pose.orientation.w
        car_theta = 2 * np.arccos(s)
        # contain theta between pi and -pi
        if car_theta > np.pi:
            car_theta = -(2 * np.pi - car_theta)
        # a big number to begin with
        mindist = 1000000

        for i in range(len(self.base_waypoints)):
            x = self.base_waypoints[i].pose.pose.position.x
            y = self.base_waypoints[i].pose.pose.position.y

            dist = math.sqrt((car_x - x) * (car_x - x) + (car_y - y) * (car_y - y))
            if (dist < mindist):
                mindist = dist
                nwp_x = x
                nwp_y = y
                nwp_index = i

        # this will be the closest waypoint index without respect to heading
        heading = np.arctan2((nwp_y - car_y), (nwp_x - car_x))
        angle = abs(car_theta - heading);
        # so if the heading of the waypoint is over one quarter of pi its behind so take the next wp :)
        if (angle > np.pi / 4):
            nwp_index = (nwp_index + 1) % len(self.base_waypoints)

        return nwp_index

    def get_next_waypoint_index2(self):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # prepare car position and orientation
        car_x = self.current_pose.position.x
        car_y = self.current_pose.position.y
        s = self.current_pose.orientation.w
        car_theta = 2 * np.arccos(s)
        # contain theta between pi and -pi
        if car_theta > np.pi:
            car_theta = -(2 * np.pi - car_theta)
        # a big number to begin with
        mindist = 1000000

        # Test for using k-d tree for quick access to nearest waypoint
        if self.waypoints_kdtree is None:
            rospy.logwarn("Waypoint Updater - Load waypoints into k-d tree")
            wp_item_list = []
            for i, wp in enumerate(self.base_waypoints):
                x = wp.pose.pose.position.x
                y = wp.pose.pose.position.y
                wp_item_list.append(Item(x, y, i))
            self.waypoints_kdtree = kdtree.create(wp_item_list)
            if DEBUG_LEVEL >= 1:
                rospy.logwarn("Waypoint Updater - Begin sample waypoints in k-d tree")
                for i, wp in enumerate(self.base_waypoints):
                    x = wp.pose.pose.position.x
                    y = wp.pose.pose.position.y
                    wp_kdtree, dist = self.waypoints_kdtree.search_nn([x, y])
                    rospy.logwarn("Waypoint Updater wp: {0:d} pos: {1:.3f},{2:.3f}".format(i, x, y))
                    rospy.logwarn("Waypoint Updater kdtree: {0:d} pos: {1:.3f},{2:.3f}".format(wp_kdtree.data.data, wp_kdtree.data.coords[0], wp_kdtree.data.coords[1]))
                    if i > 3: break
                rospy.logwarn("Waypoint Updater - End sample waypoints in k-d tree")
        wp_kdtree, dist = self.waypoints_kdtree.search_nn([self.current_pose.position.x, self.current_pose.position.y])
        nwp_x = wp_kdtree.data.coords[0]
        nwp_y = wp_kdtree.data.coords[1]
        nwp_index = wp_kdtree.data.data
        # End of test for using k-d tree for quick access to nearest waypoint

        # this will be the closest waypoint index without respect to heading
        heading = np.arctan2((nwp_y - car_y), (nwp_x - car_x))
        angle = abs(car_theta - heading);
        # so if the heading of the waypoint is over one quarter of pi its behind so take the next wp :)
        if (angle > np.pi / 4):
            nwp_index = (nwp_index + 1) % len(self.base_waypoints)

        return nwp_index

    def publish_waypoints(self, next_wp_index):
        msg = Lane()
        msg.waypoints = []
        index = next_wp_index
        for i in range(LOOKAHEAD_WPS):
            wp = Waypoint()
            wp.pose.pose.position.x = self.base_waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y = self.base_waypoints[index].pose.pose.position.y
            wp.twist.twist.linear.x = self.base_waypoints[index].twist.twist.linear.x
            msg.waypoints.append(wp)
            index = (index + 1) % len(self.base_waypoints)
        if DEBUG_LEVEL >= 2: rospy.logwarn("Waypoints published! Next Waypoint x: {0:.3f} y: {1:.3f}".format(msg.waypoints[0].pose.pose.position.x, msg.waypoints[0].pose.pose.position.y))
        self.final_waypoints_pub.publish(msg)

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
