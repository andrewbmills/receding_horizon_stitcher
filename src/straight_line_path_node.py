#!/usr/bin/env python
import sys
import numpy as np
import rospy
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

def GetPathMsg(start, goal, frame_id, stamp):
  msg = Path()
  msg.header.frame_id = frame_id
  msg.header.stamp = stamp
  pose_start = PoseStamped()
  pose_start.pose.position = start
  pose_goal = PoseStamped()
  pose_goal.pose.position = goal
  msg.poses.append(pose_start)
  msg.poses.append(pose_goal)
  return msg

class NodeManager:
  def GetGoal(self, msg):
    self.goal = msg.pose.position
    self.goal_msg = msg
    self.new_goal = True
    print("New goal at [%0.1f, %0.1f, %0.1f]" % (self.goal.x, self.goal.y, self.goal.z))
    return

  def GetStart(self, msg):
    self.start = msg.pose.pose.position
    return

  def __init__(self):
    rospy.init_node("straight_line_pather")
    rospy.Subscriber("goal", PoseStamped, self.GetGoal, queue_size=1)
    rospy.Subscriber("start", Odometry, self.GetStart, queue_size=1)
    self.pub_path = rospy.Publisher("path", Path, queue_size=1)
    self.start = Odometry()
    self.new_goal = False
    return

  def Start(self):
    rate = rospy.Rate(10.0) # Hz
    while (not rospy.is_shutdown()):
      rate.sleep()
      if (self.new_goal):
        self.pub_path.publish(GetPathMsg(self.start, self.goal, self.goal_msg.header.frame_id, self.goal_msg.header.stamp))
        self.new_goal = False
    return

if __name__ == '__main__':
	manager = NodeManager()

	try:
		manager.Start()
	except rospy.ROSInterruptException:
		pass