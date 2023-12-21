import numpy as np
import os
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy

from pyk4a import PyK4A

class PCDObsEnv:
    """
    Environment for capturing point clouds from the cameras.

    Args:
        camera_indices (list): list of camera indices to use
        camera_param_dir (str): directory containing the camera parameters
        camera_alignment (str): file containing the camera alignment. Set to False to use identity.
        voxel_size (float): voxel size for downsampling
        clip_distance (float): clip points farther than this distance
    """
    def __init__(self, 
                 camera_indices=[0, 2, 3],
                 camera_param_dir=None,
                 camera_alignments=None, 
                 voxel_size=0.005,
                 clip_distance=1.5e3):
        self.camera_indices = camera_indices
        self.voxel_size = voxel_size
        self.clip_distance = clip_distance

        # Load all the camera transforms
        if camera_param_dir is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            camera_param_dir = os.path.join(curr_dir, 'calibration/calibration_results')
        camera_params_files = {
            0: 'cam0_calibration.npz',
            2: 'cam2_calibration.npz',
            3: 'cam3_calibration.npz',
        }
        self.camera_transforms = load_camera_transforms(camera_param_dir, camera_params_files)

        # Load the default camera alignment. Load identity when set to False.
        if camera_alignments is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            camera_alignments = os.path.join(curr_dir, 'calibration/finetune_results/camera_alignments.npz')
        self.camera_alignments = load_camera_alignments(camera_alignments)

        # Start all the cameras
        self.k4as = {}
        for cam_id in camera_indices:
            try:
                k4a = PyK4A(device_id=cam_id)
                k4a.start()
                self.k4as[cam_id] = k4a
                print(f"Started camera {cam_id}")
            except:
                print(f"Failed to start camera {cam_id}")
    
    def get_pcd(self):
        """
        Get the point cloud from all the cameras.
        """
        pcds = []
        for cam_id in self.camera_indices:
            pcd = self.get_single_pcd(cam_id)
            color = np.random.rand(3)
            pcd.paint_uniform_color(color)
            if pcd is not None:
                pcds.append(pcd)
            
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            combined_pcd += pcd
        return combined_pcd
    
    def get_single_raw_pcd(self, cam_id):
        """
        Capture a single point cloud (o3d) from a camera.
        """
        k4a = self.k4as[cam_id]
        try:
            capture = k4a.get_capture()
            assert capture.depth is not None
            
            # Read pcd from k4a
            pcd = capture.depth_point_cloud
            pcd = pcd.reshape(-1, 3)

            # Clip points too far away
            distances = np.linalg.norm(pcd, axis=1)
            pcd = pcd[distances < self.clip_distance]

            pcd = pcd.astype(np.float32) / 1e3  # Convert to meters
            return pcd

        except:
            print(f"Failed to capture PCD from camera {cam_id}")
            return None
    
    def get_single_pcd(self, cam_id):
        """
        Capture a single point cloud (o3d) from a camera, 
        Perform the following steps:
            - transformed to the base frame
            - apply camera alignment
            - random downsample to 1/4 of the points
            - voxel downsample
            - remove outliers
            - estimate normals
        """
        pcd = self.get_single_raw_pcd(cam_id)
        if pcd is not None:
            # Transform to base frame
            transform = self.camera_transforms[cam_id]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            pcd = pcd.transform(transform)

            # Apply camera alignment
            pcd = pcd.transform(self.camera_alignments[cam_id])

            # Random downsample
            pcd_size = len(pcd.points)
            pcd_down_mask = np.random.choice(pcd_size, int(pcd_size / 4))
            pcd_down = pcd.select_by_index(pcd_down_mask)

            # Voxel downsample
            pcd_down = pcd_down.voxel_down_sample(self.voxel_size)

            # Remove outliers
            radius = 0.02
            nb_points = int(((radius / self.voxel_size) ** 2) * 0.8) 
            pcd_down, _ = pcd_down.remove_radius_outlier(nb_points=nb_points, radius=radius)

            # Estimate normals
            radius_normal = self.voxel_size * 4
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            return pcd_down
        else:
            return None
    
    def get_camera_coord(self, cam_id):
        """
        Get the o3d coordinate frame of a camera.
        """
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=self.camera_transforms[cam_id][:3, 3])
        coord = coord.rotate(self.camera_transforms[cam_id][:3, :3])
        return coord
    
    def get_camera_coords(self):
        """
        Get the o3d coordinate frames of all cameras.
        """
        coords = [self.get_camera_coord(cam_id) for cam_id in self.camera_indices]
        return coords
    
    def compose_transform(self, base_transforms, additional_transforms):
        """
        Compose the base transforms with the additional transforms.
        """
        new_transforms = deepcopy(base_transforms)
        for cam_id, transform in additional_transforms.items():
            new_transforms[cam_id] = new_transforms[cam_id] @ transform
        return new_transforms
    
    def visualize(self):
        """
        Visualize the point cloud from all the cameras.
        """
        combined_pcd = self.get_pcd()
        camera_coords = self.get_camera_coords()
        base_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([
            combined_pcd,
            base_coord,
            *camera_coords
            ])

def load_camera_transforms(param_dir, param_files):
    """Load the npz files containing the camera transforms"""
    camera_transforms = {}
    for cam_id, file in param_files.items():
        # Load the camera parameters
        file_path = os.path.join(param_dir, file)
        data = np.load(file_path)
        transform = data['T']

        camera_transforms[cam_id] = transform
        
    return camera_transforms

def load_camera_alignments(path):
    """Load the npz file containing the camera alignments"""
    if path is False:
        Warning(f"Camera alignment set to False. Using identity.")
        return defaultdict(lambda: np.eye(4))
    else:
        alignments = {}
        data = np.load(path)
        # The cam ids are stored as strings in the npz file, convert to int
        for cam_id, transform in data.items():
            alignments[int(cam_id)] = transform
        print(f"Loaded camera alignment for cameras {list(alignments.keys())}")    
    return alignments

# Function to load camera parameters and convert to transformation matrix
def load_camera_transforms_old(param_dir, param_files):
    t_depth_to_cam_base = np.array([0, 0, 0.0018])
    q_depth_to_cam_base = np.array([0.52548, -0.52548, 0.47315, -0.47315])
    T_depth_to_cam_base = np.eye(4)
    T_depth_to_cam_base[:3, :3] = Rotation.from_quat(q_depth_to_cam_base).as_matrix()
    T_depth_to_cam_base[:3, 3] = t_depth_to_cam_base
    
    camera_transforms = {}
    for cam_id, file in param_files.items():
        # Load the camera parameters
        file_path = os.path.join(param_dir, file)
        with open(file_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)['transformation']
            # print(params)
        
        # Convert to transformation matrix
        t = [params['x'], params['y'], params['z']]
        q = [params['qx'], params['qy'], params['qz'], params['qw']]
        rotation = Rotation.from_quat(q).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = t

        # Apply the transformation to the depth camera pose
        transform = transform @ T_depth_to_cam_base 

        camera_transforms[cam_id] = transform

    return camera_transforms

# Main program
def main():
    env = PCDObsEnv(camera_indices=[0, 2, 3])
    env.visualize()

if __name__ == "__main__":
    main()