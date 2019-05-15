#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


def talker():
	br=CvBridge()
	vid=cv2.VideoCapture("/home/jenit1/Desktop/easier.mp4")
	pub = rospy.Publisher('Lanes', Image, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(24)
	while not rospy.is_shutdown():
		_,frame=vid.read()
		if frame is None:
			break
		if cv2.waitKey(1)==27:
			pub.publish(None)
			break
		img = br.cv2_to_imgmsg(frame,'bgr8')  # Convert the image to a message
		pub.publish(img)
		rate.sleep()
if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
