import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import glob
import pickle 
from window_tracker import window_tracker

dist_p = pickle.load(open('./calibration_pickle.p', 'rb'))
mtx = dist_p["mtx"]
dist = dist_p["dist"]

def abs_sobel_thresh(img, orient = 'x', sobel_kernel=3, thresh=(0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	if orient == 'y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Find absolute gradient
	abs_sobel = np.absolute(sobel)
	# Scale to 8 bit
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Apply threshold
	grad_binary = np.zeros_like(scaled_sobel)
	grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh = (0, 255)):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Caculate gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# rescale to 8 bit
	gradmag = ((gradmag*255)/np.max(gradmag)).astype(np.uint8)
	# Apply threshold
	mag_binary = np.zeros_like(gradmag)
	mag_binary[(gradmag>= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Take sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Find absolute gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	# Find gradient direction
	direction = np.arctan2(abs_sobely, abs_sobelx)
	# Apply thresholds
	dir_binary = np.zeros_like(direction)
	dir_binary[(direction>= thresh[0]) & (direction <= thresh[1])] = 1
	
	return dir_binary

def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
	# Convert to HLS and extract S channel
	HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	s_channel = HLS[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1
	
	HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	v_channel = HSV[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

	col_binary = np.zeros_like(s_channel)
	col_binary[(s_binary == 1) & (v_binary == 1)] = 1
	
	return col_binary

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

# images = glob.glob('../CarND-Advanced-Lane-Lines/test_images/test*.jpg')
images = glob.glob('./IMG_Track1/center_2018_03_09_14_09_30_981.jpg')


for idx, fname in enumerate(images):
 	#read in image
 	img = cv2.imread(fname)
 	# Undistort each image
 	img = cv2.undistort(img, mtx, dist, None, mtx)
 	# Process image and generate binaries
 	processedImage = np.zeros_like(img[:,:,0])
 	ksize = 3
 	gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 100))
 	grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 100))
 	col_binary = color_threshold(img, sthresh = (100,255), vthresh = (50,255))
 	processedImage[((gradx == 1) & (grady == 1)) | col_binary == 1] = 255

 	# Perspective Transform
 	img_size = (img.shape[1], img.shape[0])
 	bot_width = 835./1280
 	mid_width = 453./1280 #453./1280
 	height_top = 547./720
 	height_bot = 680./720
 	w,h = img_size[0],img_size[1]
 	src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
 	offset = img_size[0]*.25
 	dst = np.float32([[(w-(.5*w))/2.,h],
                  [(w-(.5*w))/2.,0.82*h],
                  [(w+(.5*w))/2.,0.82*h],
                  [(w+(.5*w))/2.,h]])

 	M = cv2.getPerspectiveTransform(src,dst)
 	Minv = cv2.getPerspectiveTransform(dst, src)
 	warped = cv2.warpPerspective(processedImage,M,img_size,flags=cv2.INTER_LINEAR)

 	window_width = 25
 	window_height = 80
 	margin = 25 
 	smooth = 15

 	curve_points = window_tracker(window_width = window_width, window_height = window_height, margin = margin, smooth = smooth)
 	window_centroids = curve_points.find_window_centroids(warped)

 	# Points used to draw all the left and right windows
 	l_points = np.zeros_like(warped)
 	r_points = np.zeros_like(warped)

 	leftx = []
 	rightx = []
 	# Go through each level and draw the windows 
 	for level in range(0,len(window_centroids)):
 		leftx.append(window_centroids[level][0])
 		rightx.append(window_centroids[level][1])
		# Window_mask is a function to draw window areas
 		l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
 		r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		# Add graphic points from window mask here to total pixels found 
 		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
 		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	# Draw the results
 	template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
 	zero_channel = np.zeros_like(template) # create a zero color channel
 	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
 	warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
 	output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 	# Fit lane boundaries to left, right and center positions
 	yvals = range(0, warped.shape[0])
 	res_yvals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height)

 	left_fit = np.polyfit(res_yvals, leftx, 2)
 	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
 	left_fitx = np.array(left_fitx, np.int32)

 	right_fit = np.polyfit(res_yvals, rightx, 2)
 	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
 	right_fitx = np.array(right_fitx, np.int32)

 	warp_zero = np.zeros_like(warped).astype(np.uint8)
 	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
 	# yplot = np.linspace(0, 719, num = 720)
 	yplot = np.linspace(0, 159, num = 160)
 	pts_left = np.array([np.transpose(np.vstack([left_fitx, yplot]))])
 	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yplot])))])
 	pts = np.hstack((pts_left, pts_right))
 	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
 	result = cv2.addWeighted(img, 1, newwarp, 1, 0) 
 	# Define conversions in x and y from pixels space to meters
 	# ym_per_pix = 30/720 # meters per pixel in y dimension
 	# xm_per_pix = 3.7/700 # meters per pixel in x dimension
 	ym_per_pix = 30/160 # meters per pixel in y dimension
 	xm_per_pix = 3.7/320 # meters per pixel in x dimension
 	# Fit new polynomials to x,y in world space
 	left_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx, np.float32)\
 		*xm_per_pix, 2)
 	right_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(rightx, np.float32)\
 		*xm_per_pix, 2)
 	# Calculate the new radii of curvature
 	left_curverad = ((1 + (2*left_fit_cr[0]*yvals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) /\
 	 np.absolute(2*left_fit_cr[0])
 	right_curverad = ((1 + (2*right_fit_cr[0]*yvals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) /\
 	 np.absolute(2*right_fit_cr[0])
 	# Distance from road center
 	camera_center = (left_fitx[-1] + right_fitx[-1])/2
 	center_diff = (camera_center -warped.shape[1]/2)*xm_per_pix
 	side = 'left'
 	if center_diff <= 0:
 		side = 'right'

 	cv2.putText(result, 'Left Curve radius = ' +str(np.round(left_curverad,3)) + '(m)', (50,50),\
 	 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 	cv2.putText(result, 'Right Curve Radius = ' +str(np.round(right_curverad,3)) + '(m)', (50,100),\
 	 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  	
 	cv2.putText(result, str(np.round(center_diff,3)) + ' meters ' + str(side) + ' of center', (50,150),\
 	 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

 	plt.imshow(result)
 	plt.show()

 	write_name = './test_images/overlay' + str(idx) + '.jpg'
 	cv2.imwrite(write_name, result)