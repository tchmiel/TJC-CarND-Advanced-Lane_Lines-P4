## Thomas J. Chmielenski

##### P4 - Advanced Lane Finding Project
##### September 25, 2017

---
**Goals**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)



[image1]: ./output_images/chessboard.png "Chessboard Corners"
[image2]: ./output_images/orig_vs_undistorted_image.png "Original vs Undistorted Image"
[image3]: ./output_images/orig_vs_undistorted_image2.png "Original vs Undistorted Image"
[image4]: ./output_images/thresholdedx-derivative.png "Thresholded X-derivative"
[image5]: ./output_images/thresholdedy-derivative.png "Thresholded Y-derivative"
[image6]: ./output_images/thresholded_direction.png "Thresholded Magnitude"
[image7]: ./output_images/thresholded_magnitude.png "Thresholded Direction"
[image8]: ./output_images/combined_thresholds.png "Combined Thresholds"

color_binary.png
combined_thresholds.png
thresholdedx-derivative.png
thresholdedy-derivative.png
thresholded_direction.png
thresholded_magnitude.png
thresholds.png

[image116]: ./output_images/thresholds.png "Gradient Thresholds"
[image117]: ./output_images/orig_vs_combinedThresholds.png "Threshold binary Image"


[video1]: ./project_video.mp4 "Video"

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  Code for this project discussed here can be found in the IPython notebook located here: `./P4-Advanced-Lane-Lines.ipynb`


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in cells ( *CODELINE* )

I creted  a function `compute_camera_calibration_coefficents` to handle this task.

I started with `objpoints` array to store  the (x, y, z) coordinates of the chessboard 
corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z = 0, 
such that the object points are the same for each calibration image.  Thus, `objp` is just 
a replicated array of coordinates, and `objpoints` will be appended with a copy of it every 
time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with 
each successful chessboard detection.  

OpenCV gives us the `cv2.findChessboardCorners` function to  to easily find the corners in an image.  A corner is
defined as where the black and white corners meet. Coordinate 0,0,0, is at the top left.  Using the `cv2.drawChessboardCorners` method,
we can visualize this.  Here is an example:

!["Chessboard corners][image1]

Once I looped thru all 20 of the camera calibration images in the `./camera_cal/` directory., I called openCV function: `cv2.calibrateCamera` to 
return the distortion coefficients and the camera matrix that we will need to transform 3D object points
to 2D image points.  This function also returns the rotation and translations vectors for the position of
the camera in the world.

Since we have the camera matrix and distortion coefficients, we can call the `cv2.undistort` function with a distorted 
images to return us back an undistorted version of this image, also known as destination iamge.

Here is an example of the original chessboard image as well as the undistorted version:

![Original and Undistorted Images][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images).



*Thresholded X-derivative*
The `abs_sobel_thresh` function with `orient='x'` takes an image and converts it to grayscale,
It then passes this grayscale image to `cv2.Sobel` function to calculate the derivative in the 'x' direction.
This method then returns a copy of the thresholded derivative image.

The Sobel operator is at the heart of the Canny edge detection algorithm.  Applying the Sobel operator to an 
image is a way of taking the derivative of the image in the x or y direction. 

Here is an example image of Thresholded X-derivative:
![Thresholded X-derivative][image4]

*Thresholded Y-derivative*
The `abs_sobel_thresh` function with `orient='y'` is the same function above, except that it calculates
the derivative in the 'y' direciton.

Here is an example image of Thresholded Y-derivative:
!["Thresholded Y-derivative"][image5]

*Magnitude of the Gradient*
The `mag_thresh` method takes an image and converts it to grayscale,
It then passes this grayscale image to `cv2.Sobel` function to calculate both the 'x' and 'y' derivatives.
It taking the square root of the sum of each of these derivatives squared, and then 
returns a copy of the thresholded magnitude image.

Here is an example imageof Thresholded Magnitude:
!["Thresholded Magnitude"][image6]

*Direction of the Gradient*
The `dir_threshold` method is simialar to the `mag_thresh` function, but uses `arctan(sobely​​ /sobel​x)`
to determine the direction of the edges. Use a Kernel size of 15, and setting the threshold be 0.7 and 1.3 
seems reasonable to show vertical lane lines.  However it is much noiser than the gradient magnitude.

Here is an example image of Thresholded Direction:
!["Thresholded Direction"][image7]

*Combined Thresholds*
!["Combined Thresholds"][image8]


# This seciton NEEDS WORK - More explanation!

After a series of experiements, I came up with the following combined threshold:

```python
    ksize = 5 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(40, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

 


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
