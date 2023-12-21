"""
Applies o3d ICP to align pcd observations
"""

import open3d as o3d
import numpy as np
import copy

from hacman_real_env.pcd_env import PCDObsEnv

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def preprocess_point_cloud(pcd, voxel_size):
    
    print(":: Random downsample to 1/5 size.")
    pcd_size = len(pcd.points)
    pcd_down_mask = np.random.choice(pcd_size, int(pcd_size / 5))
    pcd_down = pcd.select_by_index(pcd_down_mask)

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd_down.voxel_down_sample(voxel_size)

    radius = 0.02
    nb_points = int(((radius / voxel_size) ** 2) * 0.8) 
    cl, ind = pcd_down.remove_radius_outlier(nb_points=nb_points, radius=radius)
    # cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20,
    #                                                 std_ratio=1)
    display_inlier_outlier(pcd_down, ind)
    pcd_down = cl

    radius_normal = voxel_size * 4
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def extract_features(source, target, voxel_size):
    print(":: Load two point clouds and extract_features.")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


env = PCDObsEnv()
source = env.get_single_pcd(0)
target = env.get_single_pcd(2)

# Disturb the source point cloud
# trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
trans_init = np.identity(4)
source.transform(trans_init)

# Extract features
draw_registration_result(source, target, np.identity(4))
voxel_size = 0.005
source, target, source_down, target_down, source_fpfh, target_fpfh = extract_features(
    source, target, voxel_size)

print("Apply point-to-point ICP")
threshold = 0.01
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_down, target_down, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source_down, target_down, reg_p2p.transformation)

# # Global registration
# result_ransac = execute_global_registration(source_down, target_down,
#                                             source_fpfh, target_fpfh,
#                                             voxel_size)
# print(result_ransac)
# draw_registration_result(source_down, target_down, result_ransac.transformation)

# pcd_0 = env.get_single_pcd(0)
# pcd_2 = env.get_single_pcd(2)
# pcd_3 = env.get_single_pcd(3)

# threshold = 0.02


