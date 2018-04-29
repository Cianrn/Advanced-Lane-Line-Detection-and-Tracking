
# Self-Driving Car Engineer Nanodegree
## Computer Vision
## Project - Advanced Lane Finding
***

The objective of this project is to find and track lane lines in real-time. Advanced techniques are employed in order to account for lane curvature and lighting changes. The pipeline process for this project is as follows and we will walk through each step in this report with a final demonstration on two videos:
* Camera Clibration
* Distortion Correction
* Color & Gradient Threshold
* Perspective Transform
* Identify Pixels and Fit Polynomial
* Measure Curvature

## Camera Calibration
***

Image distortion is the effect that occurs when a camera maps a 3D object onto 2D space. The transformation from 3D to 2D isnt perfect and consequently requires camera calibration to account for such distortions. The image on the top, shown below, illustrates a distorted image. The image on the bottom shows the result after distortion correction. There is noticeable differences around the endges of the board.

We know what an undostroted, flat chessboard looks like so we use this as a means of undistorting other images. We create a transform to map distorted point (`img_points`) to undistroted points (`obj_points`). Using the `cv2.calibrateCamera` we compute the camera calibration matrix and distortion coefficients and distort any images generated from this particular camera with `cv2.undistort`.

<tr>
<td> <img src="output_images/corners_found13.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/undistorted_board_example.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
</tr>

## Distortion Correction
***

Within our pipeline, we firstly undistort images based on the results above. Here is an example of image undistortion to the road landscape. The left is the original camera image and the right is undistorted.

<tr>
<td> <img src="test_images/test1.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/undistorted_road_example.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
</tr>

## Color and Gradient
***

 In order to identify lane lines, the image is processed in the following way which worked best for us. This can all be seen in `image_gen.py` attached.
* __Apply Sobel operators__: Applying these is a way of taking the derivative of an image in the x or y direction. We apply Sobel in x and y directions and identify gradients within the range (30,100). We then create a binary image. This is done using `abs_sobel_thresh`.
* __Color Thresholds__: In the HLS space, we take the s color channel and apply a threshold between (100,255). We also examine the v color channel in the HSV color space and apply a threshold between (50,255). the outcome is a binary image with ones where both s and v thresholds are satisfied. This is done using `color_threshold`. We could have also used other color channles. Further experimentation could optimize our results.
 
Below we see the processed image example. There are a number of different approaches to this. In `image_gen.py` there are other functions that we could have utilised inluding `mag_thresh` and `dir_threshold`. We can also change the hyperparamaters such as the threshold values for more accurate results.

<tr>
<td> <img src="output_images/undistorted_road_example.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/processImage0.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
</tr>

## Perspective Transform
***

In this section, we map the processed image points, shown above, to different image points to give a new perspective. This is called perspective transform. We are interested in getting a top down perspective or birds eye view which will enable us to identify lane lines and compute lane curvatures easier. We identify four points on the original image and transform those points onto our destination map. Using both `cv2.getPerspectiveTransform` and `cv2.warpPerspective`, we can create our warped image shown below on the right. Ideally, we want the gap between lanes to be even on the top and bottom of the image. In the right figure below, the distances are relatively similar.

<table>
<tr><td>**Source**</td>
<td>**Destination**</td></tr>

<tr><td>(200,720)</td>
<td>(320,720)</td></tr>

<tr><td>(453,547)</td>
<td>(320,590)</td></tr>

<tr><td>(835,547)</td>
<td>(960,590)</td></tr>

<tr><td>(1100,720)</td>
<td>(960,720)</td></tr>

<tr>
<td> <img src="output_images/processImage0.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/warped0.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
</tr>

## Identify Pixels and Fit Polynomial
***

Identifying lane pixels is done in `window_tracker.py`. We apply a convolution method. This invloves the summation of the product pf the window template and the vertical slice of the pixel image. The window is slid from left to right over the image. Overlapping values are summed. The highest overlap of pixels are the most likely position for lane lines. The position results are then appended to `window_centroids`. We continue this process for each level in the range of one to image height divided by window height. There are 9 vertical windows which can be seen in the middle figure below. 

<tr>
<td> <img src="output_images/warped0.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/windows0.jpg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
<td> <img src="output_images/lane_curvature.jpeg" alt="Drawing" style="width: 500px; height:250px" align="center"/> </td>
</tr>

## Measuring Curvature
***

The next task is to compute the lane curvature using the equations below.

$$ f_{left}(y)=A_{left}y^2 + B_{left}y + C_{left} $$

$$ f_{right}(y)=A_{right}y^2 + B_{right}y + C_{right} $$

and

$$ R_{curve} = \frac{1+(2Ay+B)^2)^{3/2}}{|2A|} $$

We also compute the vehicle distance from road center. To do this we assumed that the center of the image was the center of the vehicle and that half the distance between the first left and right polynomial points indicate where the vehicle is. The difference between these two demonstrate where the vehicle is in relation to the center of the road. This is illustarted in the immage below.

<td> <img src="output_images/overlay0.jpg" alt="Drawing" style="width: 500px; height:250px"/> </td>

## Detecting Lane Lines 
***

The videos below demonstrate our results. The initial project video on the left was relatively successful. One difficulty occured when a vehicle passed on the right. Evidently, the algorithm pivked up on this vehicle as possible lane lines and flickered a bit. 

The harder video on the right was a failure for a number of possible reasons. The roads had higher levels of curvature and we had far less forward vision.  Shadows and lighting changes were also more intense. We could potentially overcome this by altering the color and gradient section above. Another possible solution may be to look at third degree polynomials for the second, harder video. This way we might be able to model the weaving lanes better. We could also shorten how far ahead our algorithm examines given that road segments are shorter than the motorway setting in the first video.

<tr>
<td> <video controls src="output_images/output1_tracked.mp4" alt="Drawing" style="width: 475px;" align="center"/> </td>
<td> <video controls src="output_images/output2_tracked.mp4" alt="Drawing" style="width: 475px;" align="center"/> </td>
</tr>

## Conclusion
***

The project was relatively successful. There were a number of difficulties that could be fixed by further fine tuning of the algorithm and hyperparamaters. These have mostly been discussed in the report. In relation to the second video, much more work is needed.

