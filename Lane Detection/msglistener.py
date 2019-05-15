#!/usr/bin/env python
#to print the coefficients on the terminal window
import rospy
from lanes.msg import coeffs

def mytopic_callback(msg):
    print "The slope and y-intercept of the white line is :" + str(msg.white)
    print "The slope and y-intercept of the yellow line is :" + str(msg.yellow)

if __name__=='__main__':
	rospy.init_node('coefflistener', anonymous=True)
	mysub = rospy.Subscriber('coefficients', coeffs, mytopic_callback)
	rospy.spin()
