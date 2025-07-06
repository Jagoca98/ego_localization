#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

from geometry_msgs.msg import TransformStamped, Quaternion # Import TransformStamped and Quaternion
import tf_conversions # For converting numpy array to ROS Quaternion for tf broadcast
import tf2_ros # For TransformBroadcaster


from utils.ukf import BasicUKF


class EgoLocalizationNode:
    def __init__(self):
        rospy.init_node('ego_loc_node', anonymous=True)

        self.dt = rospy.get_param('~dt', 0.1)  # Default time step
        self.ukf = BasicUKF(dt=self.dt, x_init=np.ones(12))

        # Sync both topics to the same callback
        self.current_speed = None
        self.current_steering = None

        # Create subscribers
        rospy.Subscriber('/can/current_speed', Float64, self.speed_callback)
        rospy.Subscriber('/can/current_steering', Float64, self.steering_callback)
        rospy.Subscriber('/ada/insia_gps_utm_odometry', Odometry, self.gps_callback)

        # Create the publisher
        self.state_pub = rospy.Publisher('ego_loc', Odometry, queue_size=10)

        # Create tfs
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.Timer(rospy.Duration(0.1), self.odometry_callback)
        rospy.Timer(rospy.Duration(0.1), self.publish_state_callback)

        self.x_states = []
        self.p_states = []
        self.predictions = []


    def speed_callback(self, msg):
         self.current_speed = msg.data

    def steering_callback(self, msg):
         self.current_steering = msg.data

    def odometry_callback(self, event):
            if self.current_speed == None or self.current_steering == None:
                 return
            
            z = np.stack((self.current_speed / 3.6 * 1.048, self.current_steering*np.pi/180))

            # print(z)
            self.ukf.ukf.residual_z = self.ukf.residual_z_odometry 
            self.ukf.predict_ukf(fx=self.ukf.f_ca, u=z, dt=self.dt)
            self.ukf.update_ukf(z, R=self.ukf.R_odometry, hx=self.ukf.h_odometry)

    def gps_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        z = np.stack((x, y))

        self.ukf.predict_ukf(fx=self.ukf.f_same, dt=0) # process noise
        self.ukf.update_ukf(z, R=self.ukf.R_gps, hx=self.ukf.h_gps)

    def publish_state_callback(self, event):
        ego_odom = Odometry()

        # Header
        ego_odom.header.frame_id = 'map'
        ego_odom.header.stamp = rospy.Time.now()
        ego_odom.child_frame_id = 'ego_vehicle_frame'

        # Fill the position fields
        ego_odom.pose.pose.position.x = self.ukf.ukf.x[0]
        ego_odom.pose.pose.position.y = self.ukf.ukf.x[3]

        yaw = self.ukf.ukf.x[10]

        ego_odom.pose.pose.orientation.z = np.sin(yaw/2)
        ego_odom.pose.pose.orientation.w = np.cos(yaw/2)

        covariance_indices = {
            'x': 0, 'vx': 1, 'ax': 2,
            'y': 3, 'vy': 4, 'ay': 5,
            'roll': 6, 'roll_dot': 7, 'pitch': 8, 'pitch_dot': 9,
            'yaw': 10, 'yaw_dot':11
        }

        # Initialize covariance to zeros (or -1 for unknown)
        ego_odom.pose.covariance = [0.0] * 36

        ego_odom.pose.covariance[0] = self.ukf.ukf.P[covariance_indices['x'], covariance_indices['x']]
        ego_odom.pose.covariance[7] = self.ukf.ukf.P[covariance_indices['y'], covariance_indices['y']]
        ego_odom.pose.covariance[14] = 0.01 # A small constant for Z (row/col 2)
        ego_odom.pose.covariance[1] = self.ukf.ukf.P[covariance_indices['x'], covariance_indices['y']] # Cov(x,y)
        ego_odom.pose.covariance[6] = self.ukf.ukf.P[covariance_indices['y'], covariance_indices['x']] # Cov(y,x)
        ego_odom.pose.covariance[2] = 0.0 # Cov(x,z)
        ego_odom.pose.covariance[12] = 0.0 # Cov(z,x)
        ego_odom.pose.covariance[8] = 0.0 # Cov(y,z)
        ego_odom.pose.covariance[13] = 0.0 # Cov(z,y)
        ego_odom.pose.covariance[21] = self.ukf.ukf.P[covariance_indices['roll'], covariance_indices['roll']]
        ego_odom.pose.covariance[28] = self.ukf.ukf.P[covariance_indices['pitch'], covariance_indices['pitch']]
        ego_odom.pose.covariance[35] = self.ukf.ukf.P[covariance_indices['yaw'], covariance_indices['yaw']]
        ego_odom.pose.covariance[22] = self.ukf.ukf.P[covariance_indices['roll'], covariance_indices['pitch']]
        ego_odom.pose.covariance[27] = self.ukf.ukf.P[covariance_indices['pitch'], covariance_indices['roll']]
        ego_odom.pose.covariance[23] = self.ukf.ukf.P[covariance_indices['roll'], covariance_indices['yaw']]
        ego_odom.pose.covariance[33] = self.ukf.ukf.P[covariance_indices['yaw'], covariance_indices['roll']]
        ego_odom.pose.covariance[29] = self.ukf.ukf.P[covariance_indices['pitch'], covariance_indices['yaw']]
        ego_odom.pose.covariance[34] = self.ukf.ukf.P[covariance_indices['yaw'], covariance_indices['pitch']]

        # Publish odometry
        self.state_pub.publish(ego_odom)
        
        # CALCULATE QUATERNION HERE AND ASSIGN TO 'q'
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw) # Roll, Pitch, Yaw

        # --- Publish TF Transform ---
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'map' # Parent frame of the transform
        t.child_frame_id = 'ego_vehicle_frame' # Child frame of the transform

        # Set the translation from the UKF state
        t.transform.translation.x = self.ukf.ukf.x[0]
        t.transform.translation.y = self.ukf.ukf.x[3]
        t.transform.translation.z = 0.0 # Assuming 2D

        # Set the rotation (quaternion) from the UKF state
        t.transform.rotation.x = q[0] # Reuse the quaternion calculated for Odometry
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    try:
        node = EgoLocalizationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass