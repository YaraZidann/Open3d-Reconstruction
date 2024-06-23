import open3d as o3d
import numpy as np
import os
import time
# Step 1: Set the paths to RGB, depth, and camera pose files
rgb_dir = 'D:/downloads/livingroom1-color/'
# depth_dir = 'D:/downloads/NEWdepth/'
depth_dir = 'D:/downloads/ddd4epth/'
# depth_dir = 'D:/downloads/livingroom1-depth-clean/'
pose_file = 'D:/downloads/livingroom1-traj.txt'  # Assuming pose information is stored in a single file
# pose_file = 'camera_poses_ROOM1_test_v22.txt'  # Assuming pose information is stored in a single file
start = time.time()
# Step 2: Load RGB and depth files
rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')])

# Step 3: Read camera poses from the provided ground truth file
with open(pose_file, 'r') as file:
    lines = file.readlines()

# Step 4: Generate point clouds
downsampled_point_cloud = o3d.geometry.PointCloud()
previous_pcd = None

pose_idx = 0
num_images_to_process = min(len(rgb_files), 900)  # Limit the number of images to 900
print(len(rgb_files)-1)
for i in range(400,600):
    rgb_image = o3d.io.read_image(rgb_files[i])
    depth_image = o3d.io.read_image(depth_files[i])

    # Create RGB-D image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image)

    # Create camera intrinsic parameters (you may need to adjust these based on the dataset)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # Create a new point cloud from the RGB-D image and camera parameters
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Downsample the point cloud
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.0000009)  # Adjust voxel size as needed

    if previous_pcd is not None:
        # Perform registration between consecutive point clouds
        reg_p2p = o3d.pipelines.registration.registration_icp(downsampled_pcd, previous_pcd, 0.1,
                                                              np.eye(4),
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                  max_iteration=300))

        # Transform the current point cloud to the global coordinate system
        downsampled_pcd.transform(reg_p2p.transformation)

        # Add the transformed point cloud to the global point cloud
        downsampled_point_cloud += downsampled_pcd

    previous_pcd = downsampled_pcd

# Step 4: Visualize the downsampled point cloud
rotation_matrix = np.array([[1, 0, 0],
                             [0, np.cos(-np.pi/1), -np.sin(-np.pi/1)],
                             [0, np.sin(-np.pi/1), np.cos(-np.pi/1)]])

downsampled_point_cloud.rotate(rotation_matrix, center=(0, 0, 0))

# Step 5: Visualize the rotated point cloud
o3d.visualization.draw_geometries([downsampled_point_cloud])

end = time.time()
totaltime = end - start

print("Number of RGB images: ", len(rgb_files))
print("Total Run Time: ", totaltime)