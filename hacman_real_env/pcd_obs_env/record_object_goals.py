import numpy as np
import open3d as o3d
import os
import pickle
import imageio

goal_pcd_dir = os.path.join(os.path.dirname(__file__), 'goal_pcds')

def record_object_goals(num_poses=10):
    from hacman_real_env.pcd_obs_env.pcd_obs_env import PCDObsEnv
    from hacman_real_env.pcd_obs_env.segmentation import BackgroundGeometry

    obs_env = PCDObsEnv(
        voxel_size=0.002,
    )
    bg = BackgroundGeometry()
    
    # Record all the object pcds
    obj_pcds = []
    obj_imgs = []
    for _ in range(num_poses):
        frame = obs_env.record_img(crop_size=0.6)
        pcd = obs_env.get_pcd(
            return_numpy=False, color=False)
        pcd, bg_mask = bg.process_pcd(
                            pcd,
                            replace_bg=False,
                            debug=True)
        obj_pcd = pcd.select_by_index(bg_mask, invert=True)
        obj_pcds.append(obj_pcd)
        
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
    
    # Save the image previews
    n_row = 3
    n_col = len(imgs) // n_row + 1
    h, w, _ = imgs[0].shape
    preview_img = np.zeros((n_row * h, n_col * w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        row = i // n_col
        col = i % n_col
        preview_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
    preview_img_path = os.path.join(goal_pcd_dir, f'{obj_name}.png')
    imageio.imwrite(preview_img_path, preview_img)

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
    obj_name = "green_cup"
    pcds, imgs = record_object_goals(num_poses=2)
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