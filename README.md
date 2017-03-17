# Vehicle Detection Project

The goals for this project is to detect vehicles in an video coming from a camera placed in the front of a car. The following pipeline was used:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Sliding-window technique and use your trained classifier to search for vehicles in images.
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

---

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG features from the training images

To extract the hog features for the training images, I used the `skimage` hog function. Training data was composed of 64x64 images of vehicles and non-vehicles. After reading in all the images, I extracted the hog features for each image to create the training and testing datasets. This is an example of a training data for a vehicle:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

> Reference: `Bounding Box.ipynb` inside function `get_hog_features` code block `2`

#### HOG parameters.

I tried various combinations of parameters and for HOG features and compared the performance in both testing images and the testing video to get a better general feel. That is how I ended up with the following parameters:

```
spatial_size = (16, 16)
hist_bins = 32

color_space = 'YCrCb'
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'  
spatial_feat = True
hist_feat = True
hog_feat = True
```

> Reference: `Bounding Box.ipynb` code block `4`

#### Linear Classifier SVM

I trained a linear SVM using the hog and color features from the `cars` and `non-cars` vehicles dataset. This classifier will help us identify a car in patches or windows of the video frames. Before training the classifier, all the features were scaled, so as to reduce the weight of any given feature.

> Reference: `Bounding Box.ipynb` code block `5`, `8` and `9`


### Sliding Window Search

I used a sliding window search to look for cars in an image. I took different window sizes to look for vehicles in areas of the picture were it made most sense. The following window sizes were used: `64`, `96` and `128`. The search space was limited to the part of the image were it is reasonable to expect cars. In the end a total of **129** windows searched.

Using sliding windows helps search for cars throughout the whole image. Using different windows sizes lets the classifier look for cars that are at different distances, the closer the car the bigger the window we need to see it and classify it. Having different window sizes also helps by having redundancies in the classification. The image below shows the whole search space for the windows used in the sliding windows search.

![alt text][image3]

> Reference: `Bounding Box.ipynb` inside function `search_windows` code block `3`

#### Improving classifier

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also optimized the classifier by using `GridSearchCV` to find the best `C` value for the SVM (`0.5`).

---

### Video Implementation

#### Final Video
Here's a [link to my video result](./project_output.mp4).


#### Filters and False Positives

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To make detection more smooth, I implemented a moving-average of the heatmaps in the last 3 frames. This really improved the results of the video not jumping all around.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

> Reference: `Bounding Box.ipynb` for example of `14` and for detection function `16`

#### Heatmap example:
![alt text][image5]

#### Bounding box example:
![alt text][image7]


> Reference: `Bounding Box.ipynb` inside `VehicleDetection` function `detect_vehicle` in code block `16`

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


There were many challenges building the pipeline. The project depends on many parameters that were hand tuned, and are probably not generalized outside the training set. It is very time consuming to test out different feature and parameter sets.

One limitation of the current implementation is that it can be computationally heavy, given that it is doing the whole windows search for each frame.

    A way to make the model more robust is to get more training data for the classifier. Another way is to use a more sophisticated classifier, either Deep Neural Nets model or another pre-trained model.
