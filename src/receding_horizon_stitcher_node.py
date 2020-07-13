#!/usr/bin/env python
import sys
import numpy as np
from numpy import matlib
import rospy
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped


def CalculateCrosstrack(p, path):
  # Computes the n-dimensional position of the closest point to p on path
  # Inputs
  # p - (n x 1) position vector
  # path - (n x m) path list of m position vectors
  # Output
  # v - position vector of the closest point on the path
  
  (n,m) = np.shape(path)
  # Handle the 1-length paths
  if (m < 2):
    if (m == 1):
      return np.array([path[:,0]]).T, 0, 0.0
    else:
      raise Exception("path argument must be at least length 1.")

  # Calculate (x_i - x_i-1) for each path position x
  a = path[:,:m-1] # start point of each path segment
  b = path[:,1:m] # end point of each path segment
  path_diff = b-a

  # Get the length of each line segment on the path
  path_diff_len = np.linalg.norm(path_diff,axis=0)

  # Calculate the segment parameter of the closest segment point
  a_to_p = np.matlib.repmat(p, 1, m-1) - a
  t = np.sum(path_diff*a_to_p, axis=0)/(path_diff_len*path_diff_len)
  t = np.nan_to_num(t)
  

  # Make sure t values are within the range [0,1]
  t = np.minimum(np.ones(m-1), t)
  t = np.maximum(np.zeros(m-1), t)

  # Find the closest point on each segment using the parameter t
  # v = a + (b-a)*t
  v = a + path_diff*np.matlib.repmat(t, n, 1)
  i = np.argmin(np.linalg.norm(np.matlib.repmat(p, 1, m-1) - v, axis=0))
  v = np.array([v[:,i]]).T
  
  return v, i, t[i]

def CalculatePathSegmentLengths(path):
  # Calculates an m-1 long numpy array of path line segment lengths
  
  (_,m) = np.shape(path)

  # Calculate (x_i - x_i-1) for each path position x
  a = path[:,:m-1] # start point of each path segment
  b = path[:,1:m] # end point of each path segment
  path_diff = b-a # vector list from a to b

  # Get the length (2-norm) of each line segment on the path
  return np.linalg.norm(path_diff,axis=0)

def ConvertPositionToVector(position):
  v = np.zeros((3,1))
  v[0,0] = position.x
  v[1,0] = position.y
  v[2,0] = position.z
  return v

def ConvertPathToMatrix(path):
  mat = np.zeros((3, len(path.poses)))
  i = 0
  for pose in path.poses:
    mat[0,i] = pose.pose.position.x
    mat[1,i] = pose.pose.position.y
    mat[2,i] = pose.pose.position.z
    i = i + 1
  return mat

def ConvertPoseToOdometryMsg(position, frame_id, stamp, orientation=Quaternion()):
  msg = Odometry()
  msg.header.frame_id = frame_id
  msg.header.stamp = stamp
  msg.pose.pose.position.x = position[0]
  msg.pose.pose.position.y = position[1]
  msg.pose.pose.position.z = position[2]
  msg.pose.pose.orientation = orientation
  return msg

def ConvertPositionToPointStampedMsg(position, frame_id, stamp):
  msg = PointStamped()
  msg.header.frame_id = frame_id
  msg.header.stamp = stamp
  msg.point.x = position[0]
  msg.point.y = position[1]
  msg.point.z = position[2]
  return msg

def ConvertMatrixToPathMsg(mat, frame_id, stamp):
  msg = Path()
  msg.header.frame_id = frame_id
  msg.header.stamp = stamp
  (_, m) = np.shape(mat)
  for i in range(m):
    pose = PoseStamped()
    pose.pose.position.x = mat[0,i]
    pose.pose.position.y = mat[1,i]
    pose.pose.position.z = mat[2,i]
    msg.poses.append(pose)
  return msg

def FindStitchPoint(d_stitch, path, i_start, t_start):
  # Finds the cartesian coordinate position of the point d_stitch distance down the
  # path from the start point p_start = a[i_start] + (b[i_start] - a[i_start])*t_start
  #
  # Returns the final path point if d_stitch is longer than the path from the start

  (_,m) = np.shape(path)

  if (m < 2):
    if (m == 1):
      return np.array([path[:,0]]).T, 0, 0.0, Quaternion()
    else:
      raise Exception("path argument must be at least length 1.")


  # Calculate (x_i - x_i-1) for each path position x
  a = path[:,:m-1] # start point of each path segment
  b = path[:,1:m] # end point of each path segment
  path_diff = b-a # vector list from a to b

  # print("i_start = %d" % i_start)
  # print("d_stitch = %0.2f" % d_stitch)
  d = 0.0 # cumulative path length from start point
  # stitch at the end of the path if the stitch distance goes past the end
  i_stitch = m-2 # end of path segment id
  t_stitch = 1.0 # end of segment parameter
  for i in range(i_start, m-1):
    if (i == i_start):
      # Calculate the shortened first segment length
      d_segment = (1 - t_start)*np.linalg.norm(path_diff[:,i])
    else:
      # Calculate the distance from a[i] to b[i]
      d_segment = np.linalg.norm(path_diff[:,i])
    
    # print("i = %d" % (i))
    # print("d_segment = %0.2f" % (d_segment))

    # If the total path length so far is greater than the stitch distance,
    # stitch along the current segment
    if (d_stitch < (d + d_segment)):
      i_stitch = i
      if (i == i_start):
        t_stitch = t_start + (d_stitch - d)/np.linalg.norm(path_diff[:,i])
      else:
        t_stitch = (d_stitch - d)/np.linalg.norm(path_diff[:,i])
      break
    d = d + d_segment
  
  p_stitch = a[:,i_stitch] + path_diff[:,i_stitch]*t_stitch

  # Get heading of stitch segment
  yaw_stitch = np.arctan2(path_diff[1,i_stitch], path_diff[0,i_stitch])
  orientation_stitch = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw_stitch)) # 1,2,3 euler angles by default

  return np.array([p_stitch]).T, i_stitch, t_stitch, orientation_stitch

class NodeManager:
  def GetPath(self, msg):
    self.path = msg
    if (self.path.header.stamp.to_sec() == 0.0):
      self.path.header.stamp = rospy.Time.now()
    self.path_mat = ConvertPathToMatrix(msg)
    # print("Received new path!")
    return

  def GetState(self, msg):
    self.state = msg
    return
  
  def __init__(self):
    rospy.init_node("receding_horizon_stitcher")

    # Params
    self.t_horizon = rospy.get_param(rospy.get_name() + "/t_horizon", 1.0) # seconds
    self.v_estimate = rospy.get_param(rospy.get_name() + "/v_initial", 0.0) # m/s
    self.fixed_frame = rospy.get_param(rospy.get_name() + "/fixed_frame", "world")

    # Subscribers
    rospy.Subscriber("path", Path, self.GetPath, queue_size=1)
    rospy.Subscriber("odometry", Odometry, self.GetState, queue_size=1)

    # Publisher(s)
    self.pub_odometry = rospy.Publisher("odometry_stitch", Odometry, queue_size=10)
    self.pub_stitch_point = rospy.Publisher("point_stitch", PointStamped, queue_size=10)
    self.pub_path = rospy.Publisher("path_stitched", Path, queue_size=10)

    # Initialize holder variables
    self.state = Odometry()
    self.state_previous = Odometry()
    self.path = Path()
    self.path_previous = Path()
    self.path_mat = np.zeros((3,1))
    self.path_mat_previous = np.zeros((3,1))
    self.path_mat_stitched = np.zeros((3,1))
    self.path_speed_history = np.zeros(10) # Last 10 speeds
    return
  
  def EstimatePathSpeed(self):
    # Estimates the speed the vehicle is traveling, v_estimate, along the path.
    #
    p_previous = ConvertPositionToVector(self.state_previous.pose.pose.position)
    p = ConvertPositionToVector(self.state.pose.pose.position)
    (path_point_previous, _, _) = CalculateCrosstrack(p_previous, self.path_mat_stitched)
    (path_point, segment_id, t_segment) = CalculateCrosstrack(p, self.path_mat_stitched)
    speed = np.linalg.norm(path_point - path_point_previous)/(self.state.header.stamp.to_sec() - self.state_previous.header.stamp.to_sec())
    return path_point, segment_id, t_segment, speed
  
  def Start(self):
    rate = rospy.get_param(rospy.get_name() + "/rate", 10.0) # Hz
    ros_rate = rospy.Rate(10.0) # 10Hz
    while not rospy.is_shutdown():
      ros_rate.sleep()
      # Determine the current node state
      first_path_not_received = (self.path.header.stamp.to_sec() == 0.0) and (not (self.state.header.stamp.to_sec() == 0.0))
      received_new_odometry = (abs(self.state.header.stamp.to_sec() - self.state_previous.header.stamp.to_sec()) >= 1.0/rate)
      received_new_path = (abs(self.path.header.stamp.to_sec() - self.path_previous.header.stamp.to_sec()) >= 1.0/rate)

      if (first_path_not_received):
        self.path_mat_stitched[0,0] = self.state.pose.pose.position.x
        self.path_mat_stitched[1,0] = self.state.pose.pose.position.y
        self.path_mat_stitched[2,0] = self.state.pose.pose.position.z
        self.pub_odometry.publish(self.state)

      if (received_new_odometry):
        # Estimate where on the path the robot will be in t_horizon seconds
        (_, segment_id, t_segment, speed) = self.EstimatePathSpeed()
        # print("closest path point = [%0.1f, %0.1f, %0.1f]" %(path_point[0], path_point[1], path_point[2]))
        # print("segment_id = " + str(segment_id), ", t_segment = %0.2f, speed = %0.2f" % (t_segment, speed))
        self.path_speed_history = np.hstack((self.path_speed_history[1:], speed)) # Update path speed history queue
        # self.v_estimate = np.average(self.path_speed_history)
        (stitch_point, i_stitch, t_stitch, orientation_stitch) = FindStitchPoint(self.v_estimate*self.t_horizon, self.path_mat_stitched, segment_id, t_segment)
        # print("stitch_point = [%0.1f, %0.1f, %0.1f]" %(stitch_point[0], stitch_point[1], stitch_point[2]))
        # print("stitch_id = " + str(i_stitch), ", t_segment = %0.2f" % (t_stitch))
        self.pub_odometry.publish(ConvertPoseToOdometryMsg(stitch_point, self.fixed_frame, rospy.Time.now(), orientation=orientation_stitch))
        self.state_previous = self.state

      if (received_new_path):
        # Stitch the new path into previous path
        (stitch_point, i_stitch, t_stitch) = CalculateCrosstrack(np.array([self.path_mat[:,0]]).T, self.path_mat_stitched)
        (path_origin, i_origin, _, _) = FindStitchPoint(2*self.v_estimate*self.t_horizon, np.flip(self.path_mat_stitched, 1), np.size(self.path_mat_stitched, 1) - i_stitch, 1.0-t_stitch)
        i_origin = np.size(self.path_mat_stitched, 1) - i_origin - 1
        print("stitch_point: [%0.2f, %0.2f, %0.2f], i_stitch = %d, t_stitch = %0.1f" %(stitch_point[0], stitch_point[1], stitch_point[2], i_stitch, t_stitch))
        print("new path: is %d points long" % (np.size(self.path_mat,1)))
        if (np.size(self.path_mat_stitched,1) < 2):
          self.path_mat_stitched = self.path_mat
        else:
          self.path_mat_stitched = np.hstack((path_origin, self.path_mat_stitched[:,i_origin:i_stitch], stitch_point, self.path_mat))
          # self.path_mat_stitched = np.hstack((self.path_mat_stitched[:,:(i_stitch+1)], stitch_point, self.path_mat))
        print("stitched path: is %d points long" % (np.size(self.path_mat_stitched,1)))
        self.pub_path.publish(ConvertMatrixToPathMsg(self.path_mat_stitched, self.fixed_frame, rospy.Time.now()))
        self.pub_stitch_point.publish(ConvertPositionToPointStampedMsg(stitch_point, self.fixed_frame, rospy.Time.now()))
        self.path_previous = self.path
        self.path_mat_previous = self.path_mat
    return

if __name__ == '__main__':
	manager = NodeManager()

	try:
		manager.Start()
	except rospy.ROSInterruptException:
		pass
