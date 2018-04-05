## Project: Perception Pick & Place
### Writeup / README

[//]: # (Image References)

[image1]: ./imgs/Exercise1/tabletop.png
[image2]: ./imgs/Exercise1/downsampled.png
[image3]: ./imgs/Exercise1/passthrough.png
[image4]: ./imgs/Exercise1/inliers.png
[image5]: ./imgs/Exercise1/outliers.png
[image6]: ./imgs/Exercise2/point_cloud.png
[image7]: ./imgs/Exercise2/segmented_objects.png
[image8]: ./imgs/Exercise2/segmented_table.png
[image9]: ./imgs/Exercise3/normalised_confusion.png
[image10]: ./imgs/Exercise3/object_recognition.png

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You are reading it! For Exercise 1, 2 and 3, the code has been included in subfolders.

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
Please refer to [RANSAC.py](./Exercise1/RANSAC.py) for the completed code for Exercise 1. The required steps are graphically presented below:
1. Initial point cloud for tabletop (note .pcd not included in repository)

![image1]

2. Point cloud after Voxel grid downsampling

![image2]

3. Point cloud after Pass through filtering applied to the z-axis

![image3]

4. Point cloud of extracted inliers (table only)

![image4]

5. Point cloud of extracted outliers (objects only)

![image5]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
Please refer to [segmentation.py](./Exercise2/segmentation.py) for the completed code for Exercise 2. The graphical outputs from three topics are shown below:
1. Subscribed initial point cloud topic

![image6]

2. Published point cloud from pcl_objects topic after segmentation

![image7]

3. Published point cloud from pcl_table topic after segmentation

![image8]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
For the features extraction for compute_color_histograms() and compute_normal_histograms() please refer to [features.py](./Exercise3/features.py). For the color histograms, 64 bins were utilised with the range set to 0 to 256. For the normal histograms, 32 bins were utilised, with the range set to 0 to 256. For each object, 50 iterations were utilised to capture the feature vectors. The SVM was trained using a linear kernel and HSV was utilised.

The normalized confusion matrix is shown below:

![image9]

Please refer to [object_recognition.py](./Exercise3/object_recognition.py) for the completed object reg The generated model.sav file was generated, saved and loaded into the model. The output of the detected objects with their associated labels are shown below:

![image10]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



