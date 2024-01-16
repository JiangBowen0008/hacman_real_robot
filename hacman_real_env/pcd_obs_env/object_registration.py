import numpy as np
import open3d as o3d
import os
import time
import copy

from record_object_goals import load_object_goals

class ObjectRegistration():
    def __init__(self, object_name, goal_idx=None) -> None:
        self.object_name = object_name
        self.goal_pcds, self.goal_imgs = load_object_goals(object_name)
        self.resample_goal(goal_idx)
    
    def resample_goal(self, goal_idx=None):
        if goal_idx is None:
            self.goal_idx = np.random.randint(len(self.goal_pcds))
        else:
            self.goal_idx = goal_idx
    
    def get_transformed_goal_pcd(self, object_pcd, goal_idx=None):
        # Obtain the raw goal pcd
        if goal_idx is None:
            goal_idx = self.goal_idx
        raw_goal_pcd = self.goal_pcds[goal_idx]

        # Perform object registration
        goal_pcd = copy.deepcopy(raw_goal_pcd)
        transform = run_object_icp(object_pcd, goal_pcd)
        transformed_goal_pcd = copy.deepcopy(object_pcd)
        transformed_goal_pcd.transform(transform)
        return transformed_goal_pcd

    def get_goal_pcd(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.goal_idx
        return self.goal_pcds[goal_idx]
    
    def get_goal_img(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.goal_idx
        return self.goal_imgs[goal_idx]

def run_object_icp(source, target, output_fitness=False, visualize=True):
    source = copy.deepcopy(source)
    target = copy.deepcopy(target)

    t_start = time.time()
    # source = source.voxel_down_sample(0.005)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    source.orient_normals_consistent_tangent_plane(20)
    obb = source.get_oriented_bounding_box()
    obb.color = (1, 0, 0)
    
    # target = target.voxel_down_sample(0.005)
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    target.orient_normals_consistent_tangent_plane(20)
    obb_target = target.get_oriented_bounding_box()
    obb_target.color = (0, 1, 0)
    
    # Global
    voxel_size=0.005
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    ransac_transform = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    t_global = time.time()

    # source_new = copy.deepcopy(source).transform(icp_result)
    # obb_new = source_new.get_oriented_bounding_box()
    # obb_new.color = (0, 1, 0)
    # o3d.visualization.draw_geometries([source, source_new, target, obb, obb_target, obb_new])

    if visualize:
        draw_registration_result(source, target, ransac_transform)

    # Local
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=0.01, init=ransac_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=2000))
    print("ICP: Fitness {:.2f} \t MSE {:.2e}".format(reg_p2p.fitness, reg_p2p.inlier_rmse))
    icp_result = copy.deepcopy(reg_p2p.transformation)
    t_local = time.time()
    print("ICP: Global {:.2f} sec. \t Local {:.2e} sec.".format(t_global - t_start, t_local - t_global))

    print("ICP correspondences: {}".format(len(reg_p2p.correspondence_set)))

    if visualize:
        draw_registration_result(source, target, icp_result)
    
    # if reg_p2p.fitness < 0.8:
    #     print("*"*50)
    #     print("*** WARNING: ICP fitness too low! ***")
    #     print("*"*50)
    
    # source_new = copy.deepcopy(source).transform(icp_result)
    # obb_new = source_new.get_oriented_bounding_box()
    # obb_new.color = (0, 0, 1)
    # o3d.visualization.draw_geometries([source, source_new, target, obb, obb_target, obb_new])
    
    if output_fitness:
        return icp_result, reg_p2p.fitness

    return icp_result

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(20)

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    source_down = copy.deepcopy(source_down)
    target_down = copy.deepcopy(target_down)

    # Estimate translation
    source_center = np.asarray(source_down.get_center())
    target_center = np.asarray(target_down.get_center())
    translation = target_center - source_center
    source_down.translate(translation)

    distance_threshold = 0.0075
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(1),
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    transform = copy.deepcopy(result.transformation)
    translation_transform = np.eye(4)
    translation_transform[:3, 3] = translation
    transform = np.matmul(transform, translation_transform)
    return transform


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    from pcd_obs_env import PCDObsEnv
    from segmentation import BackgroundGeometry
    from record_object_goals import load_object_goals

    # Load the environment
    obs_env = PCDObsEnv()
    bg = BackgroundGeometry()
    object_reg = ObjectRegistration('blue_box')

    # Get the current object pcd
    pcd = obs_env.get_pcd(return_numpy=False)
    pcd, bg_mask = bg.process_pcd(
                        pcd,
                        replace_bg=True,
                        debug=False)
    bg_pcd = pcd.select_by_index(bg_mask)
    object_pcd = pcd.select_by_index(bg_mask, invert=True)
    # object_pcd = goals[2]

    # Perform ICP
    # goal_pcd = goals[2]
    # goal_pose = run_object_icp(object_pcd, goal_pcd, visualize=True)
    goal_idx = 8
    goal_pcd = object_reg.get_transformed_goal_pcd(object_pcd, goal_idx=goal_idx)
    gt_goal_pcd = object_reg.get_goal_pcd(goal_idx=goal_idx)

    object_pcd.paint_uniform_color([1, 0.706, 0])
    goal_pcd.paint_uniform_color([0.651, 0.929, 0])
    gt_goal_pcd.paint_uniform_color([0.2, 0.3, 0.2])
    bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([object_pcd, bg_pcd, goal_pcd, gt_goal_pcd])