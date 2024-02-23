import numpy as np
import os
import yaml
import time
import open3d as o3d
from scipy.spatial.transform import Rotation
from collections import defaultdict
from copy import deepcopy
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt

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
    camera_colors = {
        0: [1, 0.706, 0],
        1: [0, 0.651, 0.929],
        2: [0.651, 0, 0.929],
        3: [0.929, 0.651, 0],
    }   # Colors for pcd of each camera

    def __init__(self, 
                 camera_indices=[0,1,2,3],
                 camera_param_dir=None,
                 camera_alignments=None, 
                 voxel_size=0.002,
                 clip_distance=1.5):
        self.camera_indices = camera_indices
        self.voxel_size = voxel_size
        self.clip_distance = clip_distance

        # Load all the camera transforms
        if camera_param_dir is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            camera_param_dir = os.path.join(curr_dir, 'calibration/calibration_results')
        # camera_params_files = {
        #     0: 'cam0_calibration.npz',
        #     1: 'cam1_calibration.npz',
        #     2: 'cam2_calibration.npz',
        #     3: 'cam3_calibration.npz',
        # }
        camera_params_files = {
            0: 'cam0_pcd_calibration.npz',
            1: 'cam1_pcd_calibration.npz',
            2: 'cam2_pcd_calibration.npz',
            3: 'cam3_pcd_calibration.npz',
        }
        self.camera_transforms = load_camera_transforms(camera_param_dir, camera_params_files)

        # Load the default camera alignment. Load identity when set to False.
        if camera_alignments is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            camera_alignments = os.path.join(curr_dir, 'calibration/finetune_results/camera_alignments.npz')
        self.camera_alignments = load_camera_alignments(camera_alignments)

        # Start all the cameras in parallel, I/O bound
        self.k4as = {}
        with ThreadPoolExecutor() as executor:
            executor.map(self.start_camera, camera_indices)
    
    def start_camera(self, cam_id):
        try:
            k4a = PyK4A(device_id=cam_id)
            k4a.start()
            self.k4as[cam_id] = k4a
            print(f"Started camera {cam_id}")
        except:
            print(f"Failed to start camera {cam_id}")       
    
    def get_pcd(self, return_numpy=True, clip_table=True, color=False):
        """
        Get the point cloud from all the cameras. Perform voxel downsampling.
        """
        start_time = time.time()
        # Obtain pcds from all cameras in parallel, CPU bound
        pcds = []
        for cam_id in self.camera_indices:
            pcd = self.get_single_pcd(cam_id, color=color)
            
            if pcd is not None:
                pcds.append(pcd)
        
        # Combine all pcds
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            combined_pcd += pcd
        end_time = time.time()
        print(f"Obtaining pcds takes {end_time - start_time:.3f}s.")
        
        # Clip to the real table
        if clip_table:
            table_x_boundary = [0.1, 0.85]
            table_y_boundary = [-0.85, 0.25]
            combined_pcd = combined_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.array([table_x_boundary[0], table_y_boundary[0], -np.inf]),
                    max_bound=np.array([table_x_boundary[1], table_y_boundary[1], np.inf])
                )
            )
        
        # Remove outliers
        combined_pcd = combined_pcd.voxel_down_sample(0.005)
        down_pcd, idx = combined_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)
        # display_inlier_outlier(combined_pcd, idx)
        combined_pcd = down_pcd

        # Voxel downsample
        combined_pcd = combined_pcd.voxel_down_sample(self.voxel_size)
        
        # Convert back to np array
        if return_numpy:
            combined_pcd = np.asarray(combined_pcd.points)

        return combined_pcd
    
    def get_single_raw_pcd(self, cam_id, color=False):
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
            pcd = pcd.astype(np.float32) / 1e3  # Convert to meters
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))

            if color:
                colors = capture.transformed_color
                colors = colors.reshape(-1, 4)[:, :3]
                pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.0)

            # Clip points too far away
            distances = np.linalg.norm(pcd, axis=1)
            distance_mask = np.where(distances < self.clip_distance)[0]
            pcd_o3d = pcd_o3d.select_by_index(distance_mask)
            
            return pcd_o3d

        except:
            print(f"Failed to capture PCD from camera {cam_id}")
            return None
    
    def get_single_pcd(self, cam_id, color=False):
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
        start_time = time.time()
        pcd = self.get_single_raw_pcd(cam_id, color=color)
        end_time = time.time()
        if pcd is not None:
            # Transform to base frame
            transform = self.camera_transforms[cam_id]
            pcd = pcd.transform(transform)

            # Apply camera alignment
            pcd = pcd.transform(self.camera_alignments[cam_id])
            # return pcd

            # Random downsample
            pcd_size = len(pcd.points)
            pcd_down_mask = np.random.choice(pcd_size, int(pcd_size / 4))
            pcd_down = pcd.select_by_index(pcd_down_mask)

            # Voxel downsample
            voxel_size = 0.005  # For initial processing we use 0.5cm
            pcd_down = pcd_down.voxel_down_sample(voxel_size)

            # Remove outliers
            radius = 0.02
            nb_points = int(((radius / voxel_size) ** 2) * 0.95) 
            pcd_down, _ = pcd_down.remove_radius_outlier(nb_points=nb_points, radius=radius)

            if not color:
                pcd.paint_uniform_color(self.camera_colors[cam_id])
            end_time2 = time.time()
            print(f"Cam {cam_id}. Capture takes {end_time - start_time:.3f}s. Processing takes {end_time2 - end_time:.3f}s.")
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
    
    def visualize(self, clip_table=True, color=True):
        """
        Visualize the point cloud from all the cameras.
        """
        combined_pcd = self.get_pcd(return_numpy=False, clip_table=clip_table, color=True)
        combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        camera_coords = self.get_camera_coords()
        base_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([
            combined_pcd,
            base_coord,
            *camera_coords
            ])
    
    def start_video_record(self, cam_id=2):
        """
        Start a non-blocking thread that records RGB video from a camera. FPS is 30.
        """
        self.video_record_cam_id = cam_id
        self.video_frames = []
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_video)
        self.record_thread.start()

    def _record_video(self):
        """
        Internal method to record video from the camera in a separate thread.
        """
        k4a = self.k4as[self.video_record_cam_id]
        
        # Start the camera and recording process
        while self.recording:
            try:
                capture = k4a.get_capture()
                assert capture.color is not None
                frame = deepcopy(capture.color)
                self.video_frames.append(frame)
            except:
                print(f"Failed to record RGB frame from {self.video_record_cam_id}")
                return None

    def end_video_record(self):
        """
        End the video recording thread and return the recorded frames.
        Return:
            frames: list of rgb frames. FPS is 30. 
        
        Note: Color format is in BGR. Can be converted into RGB using cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).
        """
        # Stop recording and wait for the thread to finish
        self.recording = False
        self.record_thread.join()

        # Cleanup and prepare the frames to return
        del self.video_record_cam_id
        frames = deepcopy(self.video_frames)
        return frames
    
    def record_img(self, 
                   cam_id=2,
                   crop_size=1.0):
        """
        Record a single image from a camera.
        """
        k4a = self.k4as[cam_id]
        try:
            capture = k4a.get_capture()
            assert capture.color is not None
            img = deepcopy(capture.color)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Cropping from the center
            w, h = img.shape[:2]
            crop_w, crop_h = int(w * crop_size), int(h * crop_size)
            start_w, start_h = (w - crop_w) // 2, (h - crop_h) // 2
            cropped_img = img[start_w:start_w+crop_w, start_h:start_h+crop_h]

            return cropped_img
        except:
            print(f"Failed to record RGB frame from {cam_id}")
            return None

def display_inlier_outlier(cloud, ind):
    # Compute normals to help visualize
    # cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    inlier_cloud = cloud.select_by_index(ind)
    inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

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

# DEPRECATED
def load_camera_transforms_old(param_dir, param_files):
    Warning("Using deprecated function load_camera_transforms_old")
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

def save_video(frames):
    """Utility function to save the recorded video to a file."""
    import cv2
    import time
    import imageio

    # Define the codec and create VideoWriter object
    video_path = "test_video.mp4"
    writer = imageio.get_writer(video_path, fps=30) 

    for i, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)

        # import matplotlib.pyplot as plt

        # # Also save the frames as pngs
        # frame_path = f"test_video/test_video_{i}.png"
        # cv2.imwrite(frame_path, frame)

    writer.close()


# Main program
def main():
    env = PCDObsEnv(
        # camera_alignments=False,
        # camera_indices=[0,2],
        voxel_size=0.002)
    env.visualize()

    # Test camera recording
    # env.start_video_record(cam_id=2)
    # time.sleep(3)
    # frames = env.end_video_record()
    # save_video(frames)

    # Test image recording
    img = env.record_img()
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()