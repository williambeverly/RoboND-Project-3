# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')

def save_file(filename, data):
	pcl.save(data, filename)


# Voxel Grid filter
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

# Voxel Grid filter
img = voxel_grid_downsample(cloud)
save_file('voxel_downsampled.pcd',img)

# PassThrough filter
img = passthrough_filter(img, 'z', 0.6, 1.1)
save_file('pass_through_filtered.pcd',img)

# RANSAC plane segmentation
# Extract inliers
# Extract outliers
table, objects = extract_outliers_inliers(img)
# Save pcd for table
save_file('table.pcd',table)

# Save pcd for objects
save_file('objects.pcd',objects)

