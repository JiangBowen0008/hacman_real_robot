import numpy as np
import os
import yaml
from pyk4a import PyK4A
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Function to load camera parameters and convert to transformation matrix
def load_camera_transforms(param_dir, param_files):
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

        camera_transforms[cam_id] = transform
        
    return camera_transforms

def load_camera_transforms_new(param_dir, param_files):
    camera_transforms = {}
    for cam_id, file in param_files.items():
        # Load the camera parameters
        file_path = os.path.join(param_dir, file)
        data = np.load(file_path)
        R, T = data['R'], data['T']

        transform = np.eye(4)
        transform[:3, :3] = R
        quat = Rotation.from_matrix(R).as_quat()
        print(quat)
        transform[:3, 3] = T.squeeze()

        camera_transforms[cam_id] = transform
        
    return camera_transforms

# Function to capture PCD from a Kinect
def capture_pcd(camera_index, clip_distance=1.5e3):
    # Initialize the camera
    k4a = PyK4A(device_id=camera_index)
    try:
        k4a.start()
        print(f"Started camera {camera_index}")
        # extrinsics = k4a.calibration.get_extrinsic_parameters(k4a.calibration_type.COLOR, k4a.calibration_type.DEPTH)

        # Capture a single frame (implement error checking in real code)
        capture = k4a.get_capture()
        if capture.depth is not None:
            # Read pcd from k4a
            pcd = capture.depth_point_cloud
            # depth = np.linalg.norm(pcd, axis=-1)
            # plt.imshow(depth)
            # plt.show()

            pcd = pcd.reshape(-1, 3)

            # Clip points too far away
            distances = np.linalg.norm(pcd, axis=1)
            pcd = pcd[distances < clip_distance]
            # pcd = o3d.geometry.PointCloud.create_from_depth_image(
            #     o3d.geometry.Image(capture.depth),
            #     o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
            # )
            # pcd = 1e3
            pcd = pcd.astype(np.float32) / 1e3
            return pcd
        else:
            return None
            
    except:
        print(f"Failed to capture PCD from camera {camera_index}")
        return None
    
def show_pcd(pcd, orig=None, R=None, save=None, grasps=[]):
    geoms = [pcd]
    if orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=orig
        )
        if R is not None:
            coords = coords.rotate(R)
        geoms.append(coords)
    for grasp in grasps:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=grasp[:3, 3])
        coords = coords.rotate(grasp[:3, :3])
        geoms.append(coords)
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    for geom in geoms:
        viz.add_geometry(geom)
    viz.run()
    if save:
        viz.capture_screen_image(save, True)
    viz.destroy_window()

# Main program
def main():
    # Load all the camera transforms
    camera_param_dir = 'pcd_env/camera_params'
    camera_params_files = {
        # 0: 'k4a_0_eye_on_base_snapped.yaml',
        1: 'k4a_1_eye_on_base_snapped.yaml',
        # 1: 'k4a_1_eye_on_base.yaml',
        # 2: 'k4a_2_eye_on_base_snapped.yaml',
        # 3: 'k4a_3_eye_on_base_snapped.yaml'
    }
    # camera_param_dir = 'pcd_env/calibration/calibration_results'
    # camera_params_files = {
    #     # 0: 'k4a_0_eye_on_base_snapped.yaml',
    #     1: 'cam1_calibration.npz',
    #     # 1: 'k4a_1_eye_on_base.yaml',
    #     # 2: 'k4a_2_eye_on_base_snapped.yaml',
    #     # 3: 'k4a_3_eye_on_base_snapped.yaml'
    # }
    camera_transforms = load_camera_transforms(camera_param_dir, camera_params_files)

    combined_pcd = o3d.geometry.PointCloud()

    for cam_id in camera_params_files.keys():
        pcd = capture_pcd(cam_id)
        if pcd is not None:
            transform = camera_transforms[cam_id]
            # pcd = apply_transform(pcd.T, transform).T
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            transform = np.linalg.inv(transform)
            # print(transform)
            pcd = pcd.transform(transform)

            # Color the point cloud
            color = np.random.rand(3)
            # pcd.paint_uniform_color(color)
            combined_pcd += pcd
    

    # Save or visualize the combined PCD, with coordinate frame at the origin
    transformed_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=transform[:3, 3])
    transformed_coord = transformed_coord.rotate(transform[:3, :3])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries([
        combined_pcd,
        coord,
        # x_axis,
        # y_axis,
        # z_axis
        transformed_coord
        ])
    # o3d.io.write_point_cloud("combined_pcd.ply", combined_pcd)
    # o3d.visualization.draw_geometries([combined_pcd])

# Function to apply a transformation to a point cloud
def apply_transform(pcd, transform):
    r = transform[:3, :3]
    t = transform[:3, 3]
    # pcd - t[:, None]
    pcd = r @ pcd + t[:, None]
    return pcd

if __name__ == "__main__":
    main()