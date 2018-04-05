## Project: Perception Pick & Place
### Writeup / README

[//]: # (Image References)

[image1]: ./imgs/Exercise1/tabletop.png
[image2]: ./imgs/Exercise1/downsampled.png
[image3]: ./imgs/Exercise1/passthrough.png
[image4]: ./imgs/Exercise1/inliers.png
[image5]: ./imgs/Exercise1/outliers.png
[image6]: ./imgs/r3_6_calcs.png
[image7]: ./imgs/testcase_1.png
[image8]: ./imgs/display_path.png
[image9]: ./imgs/complete.png

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

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

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



