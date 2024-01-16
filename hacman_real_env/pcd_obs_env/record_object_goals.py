import numpy as np
import open3d as o3d
import os
import pickle

from pcd_obs_env import PCDObsEnv
from segmentation import BackgroundGeometry

goal_pcd_dir = os.path.join(os.path.dirname(__file__), 'goal_pcds')

def record_object_goals(num_poses=10):
    obs_env = PCDObsEnv()
    bg = BackgroundGeometry()
    
    # Record all the object pcds
    obj_pcds = []
    obj_imgs = []
    for _ in range(num_poses):
        pcd = obs_env.get_pcd(return_numpy=False)
        pcd, bg_mask = bg.process_pcd(
                            pcd,
                            replace_bg=True,
                            debug=True)
        obj_pcd = pcd.select_by_index(bg_mask, invert=True)
        obj_pcds.append(obj_pcd)

        frame = obs_env.record_img()
        obj_imgs.append(frame)
    
    return obj_pcds, obj_imgs

def save_object_goals(pcds, imgs, obj_name):
    # Convert to numpy to save the pcds
    pcds_np = []
    for pcd in pcds:
        pcd_np = np.asarray(pcd.points)
        pcds_np.append(pcd_np)
    
    # Save the pcds
    os.makedirs(goal_pcd_dir, exist_ok=True)
    file_path = os.path.join(goal_pcd_dir, f'{obj_name}.pkl')
    with open(file_path, 'wb') as f:
        content = {
            'pcds': pcds_np,
            'imgs': imgs
        }
        pickle.dump(content, f)

def load_object_goals(obj_name):
    # Load the pcds
    file_path = os.path.join(goal_pcd_dir, f'{obj_name}.pkl')
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
        pcds_np = content['pcds']
        imgs = content['imgs']
    
    # Convert to open3d
    pcds = []
    for pcd_np in pcds_np:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcds.append(pcd)
    
    return pcds, imgs


if __name__ == '__main__':
    obj_name = "blue_box"
    pcds, imgs = record_object_goals(num_poses=10)
    save_object_goals(pcds, imgs, obj_name)

    # Check the results
    pcds, imgs = load_object_goals(obj_name)

    # Plot
    import matplotlib.pyplot as plt
    n_row = 3
    n_col = len(pcds) // n_row + 1
    for i, (pcd, img) in enumerate(zip(pcds, imgs)):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()