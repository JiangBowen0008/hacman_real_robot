import open3d as o3d
import numpy as np

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

def display_pcd_segmentation(pcd, labels):
    pcds = o3d.geometry.PointCloud()
    for i in np.unique(labels):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        cluster.paint_uniform_color(np.random.rand(3))
        pcds += cluster
    o3d.visualization.draw_geometries([pcds])

def transform_point(point, transformation):
    point = np.asarray(point)
    point = np.append(point, 1)
    point = np.matmul(transformation, point)
    return point[:3]