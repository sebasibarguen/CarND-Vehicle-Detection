## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals for this project is to detect vehicles in an image, that is placed in the front of a car. 

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/HOG_example.jpg
[image3]: ./images/sliding_windows.jpg
[image4]: ./images/sliding_window.jpg
[image5]: ./images/bboxes_and_heat.png
[image6]: ./images/labels_map.png
[image7]: ./images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG features from the training images


To extract the hog features for the training images, I used the `skimage` hog function. Training data was composed of 64x64 images of vehicles and non-vehicles. After reading in all the images, I extracted the hog features for each image to create the training and testing datasets. This is an example of a training data for a vehicle:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]


#### HOG parameters.

I tried various combinations of parameters and for HOG features and compared the performance in both testing images and the testing video to get a better general feel. That is how I ended up with the following parameters:

```
spatial_size = (16, 16)
hist_bins = 32

color_space = 'HLS'
orient = 12
pix_per_cell = 8 
cell_per_block = 2
hog_channel = 'ALL'  
spatial_feat = True 
hist_feat = True 
hog_feat = True
```


#### Linear Classifier SVM

I trained a linear SVM using the hog and color features from the`cars` and `non-cars` vehicles dataset. This classifier will help us identify a car in patches or windows of the video frames. Before training the classifier, all the features were scaled, so as to reduce the weight of any given feature.



### Sliding Window Search

#### 

I used a sliding window search to look for cars. I took different window sizes to look for vehicles in areas of the picture were it made most sense. The following window sizes were used: `64`, `96` and `128`.

![alt text][image3]

#### Improving classifier 

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### Final Video
Here's a [link to my video result](./project_output.mp4).


#### Filters and False Positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


There were many challenges building the pipeline. The project depens on many parameters that were hand tuned, and are probably not generalized outside the training set. It is very time consuming to test out different feature and parameter sets.

One limitation of the current implementation is that it can be computationally heavy, given that it is doing the whole windows search for each frame. 

    A way to make the model more robust is to get more training data for the classifier. Another way is to use a more sophisticated classifier, either Deep Neural Nets model or another pre-trained model. 

