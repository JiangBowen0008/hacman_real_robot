from pyk4a import PyK4A
import matplotlib.pyplot as plt
import open3d as o3d

# Initialize the camera
k4a = PyK4A(device_id=1)
try:
    k4a.start()

    # Capture a single frame (implement error checking in real code)
    capture = k4a.get_capture()
    if capture.depth is not None:
        # Read pcd from k4a
        img = capture.color
        depth = capture.depth
        plt.imshow(img)
        plt.show()

        pcd = capture.depth_point_cloud
        pcd = pcd.reshape(-1, 3)
        print(pcd.shape)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        pcd = pcd.paint_uniform_color([1, 0.706, 0])

        transformed_pcd = capture.transformed_depth_point_cloud
        transformed_pcd = transformed_pcd.reshape(-1, 3)
        print(transformed_pcd.shape)
        transformed_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed_pcd))
        transformed_pcd = transformed_pcd.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([pcd, transformed_pcd])

    else:
        print("Failed to capture PCD from camera")
except:
    print("Camera start failed.")