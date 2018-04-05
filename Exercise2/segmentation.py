#!/usr/bin/env python

# Import modules
from pcl_helper import *

# DONE: Define functions as required
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

    # DONE: Convert ROS msg to PCL data
    img = ros_to_pcl(pcl_msg)

    # DONE: Voxel Grid Downsampling
    img = voxel_grid_downsample(img)

    # DONE: PassThrough Filter
    img = passthrough_filter(img, 'z', 0.6, 1.1)

    # DONE: RANSAC Plane Segmentation
    # DONE: Extract inliers and outliers
    table, objects = extract_outliers_inliers(img)

    # DONE: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(objects)
    cluster_indices = euclidean_clustering(white_cloud)
    
    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = visualise_clusters(cluster_indices, white_cloud)

    # DONE: Convert PCL data to ROS messages
    ros_table = pcl_to_ros(table)
    ros_objects = pcl_to_ros(cluster_cloud)

    # DONE: Publish ROS messages
    pcl_objects_pub.publish(ros_objects)
    pcl_table_pub.publish(ros_table)


if __name__ == '__main__':

    # DONE: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # DONE: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # DONE: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
