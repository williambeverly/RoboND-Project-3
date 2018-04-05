#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def voxel_grid_downsample(data, LEAF_SIZE=0.01):
    # function inputs:
    # data = point cloud in pcl format
    # LEAF_SIZE = voxel (or leaf) size
    vox = data.make_voxel_grid_filter()

    # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # return applied filter function to obtain the resultant downsampled point cloud
    return vox.filter()

def passthrough_filter(data, filter_axis, axis_min, axis_max):
    # function inputs:
    # data = point cloud in pcl format
    # filter_axis = char of 'x','y' or 'z'
    # axis_min = minimum axis value
    # axis_max = maximum axis value

    # Create a PassThrough filter object.
    passthrough = data.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)

    # Return applied filter function to obtain the resultant point cloud. 
    return passthrough.filter()

def RANSAC_plane_segmentation(data, max_distance=0.01):
    # function inputs:
    # data = point cloud data
    # max_distance = max distance for a point to be considered fitting the model
    
    # Create the segmentation object
    seg = data.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # set max distance
    seg.set_distance_threshold(max_distance)

    # Return inliners and model coefficients from segment function
    return seg.segment()

def extract_outliers_inliers(data):
    # function inputs:
    # data = point cloud data 
    # function called RANSAC_plane_segementation to identify inliers and outliers

    inliers, coefficients = RANSAC_plane_segmentation(data)
    table = data.extract(inliers, negative=False)
    objects = data.extract(inliers, negative=True)
    return table, objects

def euclidean_clustering(data, tolerance=0.02, min_size=10, max_size=10000):
    # function inputs:
    # data =  white_cloud in XYZ
    # tolerance = cluster tolerance
    # min_size = minimum cluster size
    # max_size = maximum cluster size

    tree = data.make_kdtree()

    # Create a cluster extraction object
    ec = data.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    
    # Return extracted indices for each of the discovered clusters
    return ec.Extract()

def visualise_clusters(cluster_indices, white_cloud):
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # DONE: Convert ROS msg to PCL data
    img = ros_to_pcl(pcl_msg)

    # DONE: Voxel Grid Downsampling
    img = voxel_grid_downsample(img)

    # DONE: PassThrough Filter
    img = passthrough_filter(img, 'z', 0.6, 1.1)
    # boxes are captured in first passthrough, remove these along x axis
    #img = passthrough_filter(img, 'x', 0.4, 1.1)
    
    # DONE: RANSAC Plane Segmentation

    # DONE: Extract inliers and outliers
    table, objects = extract_outliers_inliers(img)

    # DONE: Euclidean Clustering
    # create white cloud
    white_cloud = XYZRGB_to_XYZ(objects)
    cluster_indices = euclidean_clustering(white_cloud)

    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = visualise_clusters(cluster_indices, white_cloud)

    # DONE: Convert PCL data to ROS messages
    # Done in previous exercises

    # DONE: Publish ROS messages
    # Done in previous exercises

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = objects.extract(pts_list)
        # DONE: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # DONE: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)


if __name__ == '__main__':

    # DONE: ROS node initialization
    rospy.init_node('object_recog', anonymous=True)

    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # DONE: Create Publishers
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    # DONE: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # DONE: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
