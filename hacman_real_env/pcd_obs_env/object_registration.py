import numpy as np
import open3d as o3d
import os
import time
import copy

from hacman_real_env.pcd_obs_env.record_object_goals import load_object_goals

curr_dir = os.path.dirname(__file__)

class ObjectRegistration():
    def __init__(self, 
                 object_name,
                 goal_idx=None,
                 allow_full_pcd=True,
                 allow_manual_registration=False,
                 allow_approximate=True,
                 symmetric_object=False) -> None:
        self.object_name = object_name
        self.goal_pcds, self.goal_imgs = load_object_goals(object_name)
        self.full_pcd = load_object_full_pcd(object_name)
        self.goal_idx = goal_idx
        self.last_goal_idx = None
        self.allow_full_pcd = allow_full_pcd
        self.allow_manual_registration = allow_manual_registration
        self.allow_approximate = allow_approximate
        self.symmetric_object = symmetric_object
        self.resample_goal(goal_idx)
        
    def register_object(self, source, target, debug=False):
        try:
            assert not self.symmetric_object, "Symmetric object uses distance estimation!"
            reg_p2p = run_registration(
                source, target, visualize=debug)
            fitness = reg_p2p.fitness
            transform = reg_p2p.transformation
            info = {
                "fitness": reg_p2p.fitness,
                "MSE": reg_p2p.inlier_rmse,
            }
            
            # Use full pcd if fitness is low
            if (fitness < 0.98) and (self.allow_full_pcd) and (self.full_pcd is not None):
                print(":: Registering source to full pcd")
                reg_source_to_full = run_registration(source, self.full_pcd, visualize=debug)
                t_source_to_full = reg_source_to_full.transformation
                print(":: Registering target to full pcd")
                reg_target_to_full = run_registration(target, self.full_pcd, visualize=debug)
                t_target_to_full = reg_target_to_full.transformation
                t_full_to_target = np.linalg.inv(t_target_to_full)
                
                transform = t_full_to_target @ t_source_to_full
                fitness = min(reg_source_to_full.fitness, reg_target_to_full.fitness)
                rmse = reg_source_to_full.inlier_rmse + reg_target_to_full.inlier_rmse
                info = {
                    "fitness": fitness,
                    "MSE": rmse,
                }
            
            # Use manual registration if fitness is still low
            if fitness < 0.98:
                if self.allow_manual_registration:
                    reg_p2p = run_manual_registration(source, target, self.full_pcd, self.full_pcd)
                    transform = reg_p2p.transformation
                    info = {
                        "fitness": reg_p2p.fitness,
                        "MSE": reg_p2p.inlier_rmse,
                    }
                elif (fitness < 0.5) and (self.allow_approximate):
                    print(":: WARNING: ICP did not converge! Using approximate registration")
                    transform = approximate_registration(source, target, voxel_size=0.005)
                    info = {
                        "fitness": 0,
                        "MSE": 1.,
                    }
        except Exception as e:
            print(":: Object resgistration error")
            transform = approximate_registration(source, target, voxel_size=0.005)
            info = {
                "fitness": 0,
                "MSE": 1.,
            }
        
        
        return transform, info
    
    def resample_goal(self, goal_idx=None):
        self.last_goal_idx = self.goal_idx
        if goal_idx is None:
            if self.goal_idx is None:
                self.goal_idx = np.random.randint(len(self.goal_pcds))
            else:
                # Choose a different goal
                choices = list(range(len(self.goal_pcds)))
                choices.remove(self.goal_idx)
                self.goal_idx = np.random.choice(choices)
        else:
            self.goal_idx = goal_idx
    
    def get_transformed_goal_pcd(self, object_pcd, goal_idx=None, debug=False, output_info=False):
        # Obtain the raw goal pcd
        if goal_idx is None:
            goal_idx = self.goal_idx
        raw_goal_pcd = self.goal_pcds[goal_idx]

        # Perform object registration
        goal_pcd = copy.deepcopy(raw_goal_pcd)
        transform, info = self.register_object(object_pcd, goal_pcd, debug=debug)
        transformed_goal_pcd = copy.deepcopy(object_pcd)
        transformed_goal_pcd.transform(transform)
        if output_info:
            return transformed_goal_pcd, transform, info
        return transformed_goal_pcd, transform

    def get_goal_pcd(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.goal_idx
        return self.goal_pcds[goal_idx]
    
    def get_last_goal_pcd(self):
        return self.get_goal_pcd(self.last_goal_idx)
    
    def get_goal_img(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.goal_idx
        return self.goal_imgs[goal_idx]

def run_registration(source, target, output_fitness=False, visualize=False, allow_manual_registration=False):
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
    
    # Local
    best_result = (0, None)
    for i in range(8):
        ransac_transform = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        t_global = time.time()

        # source_new = copy.deepcopy(source).transform(icp_result)
        # obb_new = source_new.get_oriented_bounding_box()
        # obb_new.color = (0, 1, 0)
        # o3d.visualization.draw_geometries([source, source_new, target, obb, obb_target, obb_new])

        if visualize:
            draw_registration_result(source_down, target_down, ransac_transform)

        reg_p2p = run_object_icp(
            source_down, target_down, ransac_transform,
            # max_correspondence_distance=0.004,
            )
        print("ICP: Fitness {:.2f} \t MSE {:.2e}".format(reg_p2p.fitness, reg_p2p.inlier_rmse))
        # icp_result = copy.deepcopy(reg_p2p.transformation)

        t_local = time.time()
        print("ICP: Global {:.2f} sec. \t Local {:.2e} sec.".format(t_global - t_start, t_local - t_global))
        print("ICP correspondences: {}".format(len(reg_p2p.correspondence_set)))
        if reg_p2p.fitness > best_result[0]:
            best_result = (reg_p2p.fitness, reg_p2p)
        if reg_p2p.fitness > 0.98:
            break
    
    fitness, reg_p2p = best_result
    if fitness < 0.98:
        print("*"*50)
        print("*** WARNING: ICP did not converge! ***")
        print("*"*50)

        # Manual Registration
        # if allow_manual_registration:
        #     print("Manual registration")
        #     reg_p2p = run_manual_registration(source, target)
        #     icp_result, fitness = reg_p2p.transformation, reg_p2p.fitness
        #     visualize = True
    
    if visualize:
        draw_registration_result(source, target, reg_p2p.transformation)
    
    return reg_p2p

def pick_points(pcd1, pcd2, tansform1):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    pcd1 = copy.deepcopy(pcd1)
    pcd2 = copy.deepcopy(pcd2)
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    pcd1.transform(tansform1)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd1 + pcd2)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    picked_points = vis.get_picked_points()
    num_points = len(picked_points)
    assert num_points % 2 == 0, "Must pick an even number of points"

    if num_points == 0:
        return None
    
    picked_points1 = picked_points[:num_points // 2]
    picked_points2 = np.array(picked_points[num_points // 2:]) - len(pcd1.points)
    return picked_points1, picked_points2


def run_manual_registration(source, target, source_bg=None, target_bg=None):
    print(":: Performing manual registration")
    print(":: Visualization of two point clouds before manual alignment")
    trans_init = np.identity(4)
    picked_id_source = []
    picked_id_target = []

    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        source.orient_normals_consistent_tangent_plane(20)
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        target.orient_normals_consistent_tangent_plane(20)

    while True:
        # pick points from two point clouds and builds correspondences
        picked_ids = pick_points(source, target, trans_init)
        if picked_ids is None:
            break

        picked_id_source.extend(picked_ids[0])
        picked_id_target.extend(picked_ids[1])
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                                                o3d.utility.Vector2iVector(corr))

        # point-to-point ICP for refinement
        print(":: Performing PCD ICP refinement")
        reg_p2p = run_object_icp(source, target, trans_init)
        trans_init = copy.deepcopy(reg_p2p.transformation)
    
    return reg_p2p


def run_object_icp(source, target, init_transform, max_correspondence_distance=0.01):
    reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=max_correspondence_distance, init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-06, relative_rmse=1e-06, max_iteration=2000))
    return reg_p2p

def approximate_registration(source, target, voxel_size):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    source_aabb = source_down.get_axis_aligned_bounding_box()
    target_aabb = target_down.get_axis_aligned_bounding_box()
    source_centroid = np.asarray(source_aabb.get_center())
    target_centroid = np.asarray(target_aabb.get_center())
    transform = np.eye(4)
    transform[:3, 3] = (target_centroid - source_centroid)
    return transform

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

    distance_threshold = 0.015
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(1.2),
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    transform = copy.deepcopy(result.transformation)
    translation_transform = np.eye(4)
    translation_transform[:3, 3] = translation
    transform = np.matmul(transform, translation_transform)
    return transform

def load_object_full_pcd(object_name, scan_dir='full_pcds'):
    full_pcd_dir = os.path.join(curr_dir, scan_dir)
    if scan_dir == 'full_pcds':
        file_path = os.path.join(full_pcd_dir, f'{object_name}.pcd')
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            return None
        pcd = o3d.io.read_point_cloud(file_path)
    elif scan_dir == 'object_scans':
        # Load from .obj files in object_scans
        file_path = os.path.join(full_pcd_dir, f'{object_name}.obj')
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            return None
        pcd = o3d.io.read_triangle_mesh(file_path)
        pcd = pcd.sample_points_uniformly(number_of_points=10000)
        pcd = pcd.voxel_down_sample(voxel_size=0.001)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


"""
Test script
"""
def clip_gripper_pcd(pcd, gripper_z):
    pcd_z = np.asarray(pcd.points)[:, 2]
    gripper_mask = np.where(pcd_z > gripper_z)[0]
    object_pcd = pcd.select_by_index(gripper_mask, invert=True)
    return object_pcd

def benchmark_registration(
        object_name,
        num_trials=10):
    from pcd_obs_env import PCDObsEnv
    from segmentation import BackgroundGeometry
    from record_object_goals import load_object_goals

    # Load the environment

    obs_env = PCDObsEnv()
    bg = BackgroundGeometry()
    object_reg = ObjectRegistration(
        object_name,
        allow_manual_registration=True,
        )

    # Get the current object pcd
    from tqdm import tqdm
    for i in tqdm(range(num_trials)):
        pcd = obs_env.get_pcd(return_numpy=False)
        pcd, bg_mask = bg.process_pcd(
                            pcd,
                            replace_bg=True,
                            debug=False)
        bg_pcd = pcd.select_by_index(bg_mask)
        object_pcd = pcd.select_by_index(bg_mask, invert=True)
        # object_pcd = goals[2]

        # Perform ICP
        goal_pcd, transform, info = object_reg.get_transformed_goal_pcd(
            object_pcd,
            # debug=True,
            output_info=True,
            )
        gt_goal_pcd = object_reg.get_goal_pcd()

        object_pcd.paint_uniform_color([1, 0.706, 0])
        goal_pcd.paint_uniform_color([0.651, 0.929, 0])
        gt_goal_pcd.paint_uniform_color([0.2, 0.3, 0.2])
        bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # Save the screenshot to a file
        dir_path = os.path.join(curr_dir, f'registration_benchmark_{object_name}')
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{object_name}_{i}_{np.round(info['fitness'], 4)}_{np.round(info['MSE'], 6)}.png")

        pcd = object_pcd + bg_pcd + goal_pcd + gt_goal_pcd
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path)
        vis.destroy_window()

        object_reg.resample_goal()

def test_single_object(object_name):
    from pcd_obs_env import PCDObsEnv
    from segmentation import BackgroundGeometry
    from record_object_goals import load_object_goals

    # Load the environment
    obs_env = PCDObsEnv()
    bg = BackgroundGeometry()
    object_reg = ObjectRegistration(
        object_name,
        # allow_manual_registration=True,
        )

    # Get the current object pcd
    for i in range(3):
        pcd = obs_env.get_pcd(return_numpy=False)
        pcd, bg_mask = bg.process_pcd(
                            pcd,
                            replace_bg=True,
                            debug=False)
        bg_pcd = pcd.select_by_index(bg_mask)
        object_pcd = pcd.select_by_index(bg_mask, invert=True)
        object_pcd = clip_gripper_pcd(object_pcd, 0.26)
        # object_pcd = goals[2]

        # Perform ICP
        # goal_pcd = goals[2]
        # goal_pose = run_object_icp(object_pcd, goal_pcd, visualize=True)
        # goal_idx = 8
        goal_pcd, transform = object_reg.get_transformed_goal_pcd(
            object_pcd,
            debug=True
            )
        gt_goal_pcd = object_reg.get_goal_pcd()

        object_pcd.paint_uniform_color([1, 0.706, 0])
        goal_pcd.paint_uniform_color([0.651, 0.929, 0])
        gt_goal_pcd.paint_uniform_color([0.2, 0.3, 0.2])
        bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([object_pcd, bg_pcd, goal_pcd, gt_goal_pcd])

if __name__ == '__main__':
    # benchmark_registration('rubiks_cube', num_trials=20)
    test_single_object('white_box')
    