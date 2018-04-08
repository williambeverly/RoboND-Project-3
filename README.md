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
[image11]: ./imgs/3DPerception/pick1.png
[image12]: ./imgs/3DPerception/pick2.png
[image13]: ./imgs/3DPerception/pick3.png
[image14]: ./imgs/3DPerception/pick3_b.png

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
For the features extraction for compute_color_histograms() and compute_normal_histograms() please refer to [features.py](./Exercise3/features.py). For the color histograms, 64 bins were utilised with the range set to 0 to 256. For the normal histograms, 32 bins were utilised, with the range set to -1.0 to 1.0. For each object, 50 iterations were utilised to capture the feature vectors. The SVM was trained using a sigmoid kernel and HSV was utilised and [capture_features.py](./Exercise3/capture_features.py) has been included for reference purposes.

The normalized confusion matrix is shown below:

![image9]

Please refer to [object_recognition.py](./Exercise3/object_recognition.py) for the completed object recognition implementation. The model.sav file was generated, saved and loaded into the model. The output of the detected objects with their associated labels are shown below:

![image10]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

For the provided output_x.yaml files, please see [output_1.yaml](./output_1.yaml), [output_2.yaml](./output_2.yaml) and [output_3.yaml](./output_3.yaml). For the 3D perception pipeline, please refer to [project_template.py](./project_template.py). The model.sav was trained on the objects in world_3 and utilised across all worlds for detection. The following images demonstrate the effectiveness of the pipeline.

1. World 1 (3/3 objects detected)

![image11]

2. World 2 (4/5 objects detected)

![image12]

3. World 3 (6/8 objects detected)

![image13]

As well as the top view for World 3, as the labels are starting to become difficult to distinguish from the front view.

![image14]

### Changes for second submission

In my first submission, I noted that utilising normals histogram function and using the correct range of (-1.0, 1.0) worsened my detection. Therefore, I changed the range back to (0, 256) and had 100% detection in my pick worlds. Please note that in my second submission, I have changed it back to (-1.0, 1.0). Previously, when utilising the range of (0, 256) for my normals, it ignored the negative values, and also binned all values into a single bin (as they ranged between 0 to 1). As a result, the normal histograms were not providing information that was helpful in training the SVM to classify objects. Therefore, I assume that that the classification of objects by the SVM was dependent on colour.

By introducing the normals histogram, the detection of objects changed from 100% in all worlds to 100%, 80% and 75% in worlds 1, 2 and 3 respectively, therefore worsening my detection. I also read references to this on the perception Slack channel. Therefore, the object classification seemed to be better based only on colour. Therefore, I assume that the randomly generated rotation of objects in the training set causes more variation in the normals histogram distribution across x, y and z, than the colour histogram. In this case, colour should be more indepedent of rotation, given that each object is fairly uniform in colour. I assume that the SVM was unable to accurately utilise the normals histogram data to increase the fit on the data. It could be that the SVM hyper-parameters need to be further tweaked, or that I could attempt to decouple the rotation of the object through other techniques.

### Conclusion

This concludes the project for a passing submission. As a note to myself, I had to pass through filter in the x-axis (in addition to z) to trim out the side boxes, as these were being detected.
