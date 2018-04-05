#!/usr/bin/env python

# Import modules
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

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

# global values for setting scene and output file
SCENE_NUMBER = 1
OUTPUT_FILENAME = "output_{}.yaml".format(str(SCENE_NUMBER))


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


def statistical_outlier_filtering(data, k=20, std_dev=0.5):
    # function inputs:
    # data = point cloud in pcl format
    # k = number neighbouring points to analyse
    # std_dev = distance outside which point considered outlier

    # create filter object
    outlier_filter = data.make_statistical_outlier_filter()

    # set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(std_dev)

    # return the filter function
    return outlier_filter.filter()

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

def detect_objects(cluster_indices, objects, white_cloud):
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = objects.extract(pts_list)
        # DONE: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # DONE: complete this step just as is covered in capture_features.py
        # Extract histogram features
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
    return detected_objects, detected_objects_labels

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # DONE: Convert ROS msg to PCL data
    img = ros_to_pcl(pcl_msg)
    
    # DONE: Statistical Outlier Filtering
    img = statistical_outlier_filtering(img)

    # DONE: Voxel Grid Downsampling
    img = voxel_grid_downsample(img)

    # DONE: PassThrough Filter
    img = passthrough_filter(img, 'z', 0.6, 1.1)
    # boxes are captured in first passthrough, remove these along x axis
    img = passthrough_filter(img, 'x', 0.4, 1.1)

    # DONE: RANSAC Plane Segmentation

    # DONE: Extract inliers and outliers
    table, objects = extract_outliers_inliers(img)

    # create white cloud
    white_cloud = XYZRGB_to_XYZ(objects)

    # DONE: Euclidean Clustering
    cluster_indices = euclidean_clustering(white_cloud)

    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = visualise_clusters(cluster_indices, white_cloud)

    # DONE: Convert PCL data to ROS messages
    ros_objects = pcl_to_ros(objects)
    ros_table = pcl_to_ros(table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # DONE: Publish ROS messages
    pcl_objects_pub.publish(ros_objects)
    pcl_table_pub.publish(ros_table)
    pcl_cluster.publish(ros_cluster_cloud)

    # Detect objects and object labels
    detected_objects, detected_objects_labels = detect_objects(cluster_indices, objects, white_cloud)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service

def pr2_mover(object_list):

    # DONE: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = SCENE_NUMBER
    yaml_output = []

    # DONE: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')

    # DONE: Loop through the pick list
    for object in object_list_param:

        object_name = String()
        arm_name = String()
        pick_pose = Pose()
        place_pose = Pose()

        object_name.data = object['name']

        # DONE: Assign the arm to be used for pick_place
        # DONE: Create 'place_pose' for the object and assign

        if(object['group'] == 'red'):
            arm_name.data = 'left'
            place_pose.position.x = np.float(dropbox_list_param[0]['position'][0])
            place_pose.position.y = np.float(dropbox_list_param[0]['position'][1])
            place_pose.position.z = np.float(dropbox_list_param[0]['position'][2])
        else:
            arm_name.data = 'right'
            place_pose.position.x = np.float(dropbox_list_param[1]['position'][0])
            place_pose.position.y = np.float(dropbox_list_param[1]['position'][1])
            place_pose.position.z = np.float(dropbox_list_param[1]['position'][2])

        # find a matching object
        matching_object = None
        for match in object_list:
            if match.label == object['name']:
                matching_object = match
                break

        if matching_object != None:
            # DONE: Get the PointCloud for a given object and obtain it's centroid
            points_arr = ros_to_pcl(matching_object.cloud).to_array()
            x, y, z = np.mean(points_arr, axis=0)[:3]
            pick_pose.position.x = np.asscalar(x)
            pick_pose.position.y = np.asscalar(y)
            pick_pose.position.z = np.asscalar(z)

            # DONE: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            # add the data to yaml_output
            yaml_output.append(yaml_dict)
        else:
            # when duplicate objects are detected, make this fail gently
            print("Object not identified correctly")

        # Comment out the routine to just write the yaml files
        '''
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''

    # DONE: Output your request parameters into output yaml file
    send_to_yaml(OUTPUT_FILENAME, yaml_output)

if __name__ == '__main__':

    # DONE: ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # DONE: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", pc2.PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", pc2.PointCloud2, queue_size=1)
    pcl_cluster = rospy.Publisher("/pcl_cluster", pc2.PointCloud2, queue_size=1)
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
