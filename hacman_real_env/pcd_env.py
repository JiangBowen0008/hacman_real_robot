import numpy as np
import os
import yaml
from pyk4a import PyK4A
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

t_depth_to_cam_base = np.array([0, 0, 0.0018])
q_depth_to_cam_base = np.array([0.52548, -0.52548, 0.47315, -0.47315])
T_depth_to_cam_base = np.eye(4)
T_depth_to_cam_base[:3, :3] = Rotation.from_quat(q_depth_to_cam_base).as_matrix()
T_depth_to_cam_base[:3, 3] = t_depth_to_cam_base

class PCDObsEnv:
    def __init__(self, camera_indices=[0, 2, 3], clip_distance=1.5e3):
        self.camera_indices = camera_indices
        self.clip_distance = clip_distance

        # Load all the camera transforms
        camera_param_dir = 'pcd_env/calibration/calibration_results'
        camera_params_files = {
            0: 'cam0_calibration.npz',
            2: 'cam2_calibration.npz',
            3: 'cam3_calibration.npz',
        }
        self.camera_transforms = load_camera_transforms(camera_param_dir, camera_params_files)

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
        Capture a single point cloud (o3d) from a camera, transformed to the base frame.
        """
        pcd = self.get_single_raw_pcd(cam_id)
        if pcd is not None:
            transform = self.camera_transforms[cam_id]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            pcd = pcd.transform(transform)
            return pcd
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
    camera_transforms = {}
    for cam_id, file in param_files.items():
        # Load the camera parameters
        file_path = os.path.join(param_dir, file)
        data = np.load(file_path)
        transform = data['T']
        # R, T = data['R'], data['T']

        # transform = np.eye(4)
        # transform[:3, :3] = R
        # quat = Rotation.from_matrix(R).as_quat()
        # print(quat)
        # transform[:3, 3] = T.squeeze()

        camera_transforms[cam_id] = transform
        
    return camera_transforms

# Function to load camera parameters and convert to transformation matrix
def load_camera_transforms_old(param_dir, param_files):
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