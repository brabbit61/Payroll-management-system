import cv2
import numpy as np
framenum=0 
cordpos=[0,0]
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
name=raw_input("Enter the video name: ")
vid=cv2.VideoCapture(name+".mp4")
while vid.isOpened():
	framenum+=1
	_,ogimg=vid.read()
	if ogimg is None:
		break
	w,h,_=ogimg.shape										#dimensions of the img
	'''
	M=cv2.getRotationMatrix2D((w/2,h/2),270,1)
	ogimg=cv2.warpAffine(ogimg,M,(h,w))
	'''
	finaly=np.zeros((w,h,3),np.uint8)						#black background for yellow color detection
	finalw=np.zeros((w,h,3),np.uint8)						#black background for white color detection
	white_lower=(206,195,181)
	white_upper=(255,255,255)
	trapezium=np.array([[150,680],[1150,680],[900,440],[270,440]])	#ROI
	mask=np.zeros((w,h,3),np.uint8)
	cv2.fillPoly(mask,[trapezium],(255,255,255))
	img=cv2.bitwise_and(ogimg,mask)
	
	#for white lines
	maskw = cv2.inRange(img, white_lower,white_upper)		#mask for white color
	finalw=cv2.cvtColor(finalw,cv2.COLOR_BGR2GRAY)			#grayscale for the detected white color
	finalw=cv2.GaussianBlur(maskw,(3,3),0)
	frame= cv2.Canny(finalw,50,150,L2gradient=True)			#Find the outlines of the white lines on the road
	frame=cv2.dilate(frame,None,iterations=3)
	frame=cv2.erode(frame,None,iterations=1)
	lines = cv2.HoughLinesP(frame, 1, np.pi/180,175,minLineLength=60,maxLineGap=200)														
	mw=0
	isfirstline=True
	if lines is not None:
		slopepos=[]
		intercepts=[]
		for x in lines:
			for y in x:
				if y[2]-y[0] ==0:
					break
				mw=(y[3]-y[1])/float(y[2]-y[0])
			 	if mw>0.35:
					cw=y[1]-mw*y[0]
					intercepts.append(cw)
					slopepos.append(mw)
					dist=abs(cordpos[0]-y[0])+abs(cordpos[1]-y[1])
					if dist<30:
						continue
					cordpos=[y[0],y[1]]
		
		if len(slopepos)>0:
			cw=sum(intercepts)/len(intercepts)				# y intercept
			mw=sum(slopepos)/len(slopepos)					#Slope for the white lines		
			mw=np.arctan(mw)
			if abs(mw)>0.35:											#to exclude lines that have very low degree of inclination eg 1,2 degrees			
				x2= int(cordpos[0]+285*np.cos(mw))
				y2= int(cordpos[1]+285*np.sin(mw))
				cv2.line(ogimg,(cordpos[0],cordpos[1]),(x2,y2),(0,0,255),9)
				isfirstline=False
				
	else:
		if not isfirstline:
			cv2.line(ogimg,(cordpos[0],cordpos[1]),(x2,y2),(0,0,255),9)
			
	
	#for the yellow lines on the road
	framey=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)				#detection of yellow color
	yellow_lower=(15,95,13)								#threshold values
	yellow_upper=(34,180,129)
	maskydark = cv2.inRange(framey, yellow_lower,yellow_upper)	#mask for yellow color for poorly light areas
	finaly=cv2.GaussianBlur(maskydark,(3,3),0)
	lol= cv2.Sobel(finaly,cv2.CV_8U,1,0,ksize=5)
	frame= cv2.Canny(lol,50,150,L2gradient=True)			#Find the outlines of the yellow lines on the road
	frame=cv2.dilate(frame,None,iterations=2)
	frame=cv2.erode(frame,None,iterations=1)
	_,contours2,heirachy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	#finding contours in the image
	if contours2 is not None:
		area=0
		for cnt in contours2:
			x,y,w,h = cv2.boundingRect(cnt)			
			M=cv2.moments(cnt)
			m=-h/float(w)									#slope of the yellow line		
			c=y+h-m*(x)										#intercept of the yellow line
			#if m<-0.59 and m>-1.3:
			if abs(m)<1.3:	
				if M['m00']>area:
					area=M['m00']	
		for cnt in contours2:
			M=cv2.moments(cnt)
			if M['m00']==area:		
				rect=cv2.minAreaRect(cnt)
				x,y,w,h = cv2.boundingRect(cnt)		
				if ((rect[1][1]*2.8)<rect[1][0]) or ((rect[1][0]*2.8)<rect[1][1]):	#checking whether its a line or not 
					m=-h/float(w)
					cv2.line(ogimg,(x,y+h),(x+w,y),(0,0,255),9)	
				break
	cv2.imshow('frame',ogimg)
	if cv2.waitKey(1)==27:
		break
cv2.destroyAllWindows()
