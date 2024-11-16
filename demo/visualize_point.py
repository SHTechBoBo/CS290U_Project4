import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))

from model import Model

def vis_kp(left_joint_coords,
            right_joint_coords,
            save_path=None):
    # 3D Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Joint coordinates
    left_x = left_joint_coords[:, 0]
    left_y = left_joint_coords[:, 1]
    left_z = left_joint_coords[:, 2]
    
    right_x = right_joint_coords[:, 0]
    right_y = right_joint_coords[:, 1]
    right_z = right_joint_coords[:, 2]
    
    # Scatter plot for keypoints
    ax.scatter(left_x, left_y, left_z, c="r", marker='o', label='Left hand')
    ax.scatter(right_x, right_y, right_z, c="b", marker='o', label='Right hand')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=45, azim=45)
    ax.legend()
    plt.savefig(save_path)
    plt.close(fig)
    
    # 2D Visualizations
    # XY Plane
    fig_xy = plt.figure()
    plt.scatter(left_x, left_y, c="r", label="Left hand")
    plt.scatter(right_x, right_y, c="b", label="Right hand")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY Plane")
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_XY.png")
    plt.close(fig_xy)
    
    # YZ Plane
    fig_yz = plt.figure()
    plt.scatter(left_y, left_z, c="r", label="Left hand")
    plt.scatter(right_y, right_z, c="b", label="Right hand")
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.title("YZ Plane")
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_YZ.png")
    plt.close(fig_yz)
    
    # ZX Plane
    fig_zx = plt.figure()
    plt.scatter(left_z, left_x, c="r", label="Left hand")
    plt.scatter(right_z, right_x, c="b", label="Right hand")
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.title("ZX Plane")
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}_ZX.png")
    plt.close(fig_zx)

def main():
    test_folder = './imgs'
    output_dict = './output_kp'
    model_path = './snapshot_99.pth.tar'
    model = Model(resnet_version=50, mano_neurons=[512, 512, 512, 512], mano_use_pca=False, cascaded_num=3)
    # device_run = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_run = torch.device("cpu")
    ckpt = torch.load("{}".format(model_path))
    model.load_state_dict({k.split('.', 1)[1]: v for k, v in ckpt['network'].items()})
    model.to(device_run)
    model.eval()
    print('load success')
    INPUT_SIZE = 256
    right_face = model.mesh_reg.mano_layer['r'].faces
    left_face = model.mesh_reg.mano_layer['l'].faces
    for img_name in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, img_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        ratio = INPUT_SIZE / max(*img.shape[:2])
        M = np.array([[ratio, 0, 0], [0, ratio, 0]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR, borderValue=[0, 0, 0])
        img = img[:, :, ::-1].astype(np.float32) / 255
        input_tensor = torch.tensor(img.copy().transpose(2, 0, 1), device=device_run, dtype=torch.float32).unsqueeze(0)
        out = model({'img': input_tensor}, None, None, 'test')

        joint_coord_out = out['joints3d']
        trans = out['trans']
        verts3d = out['verts3d']
        right_mano_para = {
            'joints3d': joint_coord_out[:, :21, :] - joint_coord_out[:, 4, None, :],
            'verts3d': verts3d[:, :verts3d.shape[1] // 2, :] - joint_coord_out[:, 4, None, :],
        }
        left_mano_para = {
            'joints3d': joint_coord_out[:, 21:, :] - joint_coord_out[:, 4 + 21, None, :],
            'verts3d': verts3d[:, verts3d.shape[1] // 2:, :] - joint_coord_out[:, 4 + 21, None, :],
        }

        predict_right_length = (right_mano_para['joints3d'][:, 4] - right_mano_para['joints3d'][:, 0]).norm(dim=1)
        predict_left_length = (left_mano_para['joints3d'][:, 4] - left_mano_para['joints3d'][:, 0]).norm(dim=1)
        predict_right_verts = right_mano_para['verts3d'] / predict_right_length[:, None, None]
        predict_left_verts = left_mano_para['verts3d'] / predict_left_length[:, None, None]

        predict_left_verts_trans = (predict_left_verts + trans[:, 1:].view(-1, 1, 3)) * torch.exp(trans[:, 0, None, None])

        output_file_name = img_name.split('.')[0]

        # Visualize the 3D & 2D keypoints
        vis_kp(left_mano_para['joints3d'][0].cpu().numpy(),
                               right_mano_para['joints3d'][0].cpu().numpy(), 
                               save_path=os.path.join(output_dict, output_file_name + '_keypoints.png'))


        # Save the meshes as .obj files
        with open(os.path.join(test_folder, output_file_name + '_right.obj'), 'w') as file_object:
            for v in predict_right_verts[0]:
                print("v %f %f %f" % (v[0], v[1], v[2]), file=file_object)
            for f in right_face + 1:
                print("f %d %d %d" % (f[0], f[1], f[2]), file=file_object)

        with open(os.path.join(test_folder, output_file_name + '_left.obj'), 'w') as file_object:
            for v in predict_left_verts_trans[0]:
                print("v %f %f %f" % (v[0], v[1], v[2]), file=file_object)
            for f in left_face + 1:
                print("f %d %d %d" % (f[0], f[1], f[2]), file=file_object)

        with open(os.path.join(test_folder, output_file_name + '_interacting.obj'), 'w') as file_object:
            for v in predict_right_verts[0]:
                print("v %f %f %f" % (v[0], v[1], v[2]), file=file_object)
            for v in predict_left_verts_trans[0]:
                print("v %f %f %f" % (v[0], v[1], v[2]), file=file_object)
            for f in right_face + 1:
                print("f %d %d %d" % (f[0], f[1], f[2]), file=file_object)
            for f in left_face + 1 + predict_right_verts.shape[1]:
                print("f %d %d %d" % (f[0], f[1], f[2]), file=file_object)
                
        
if __name__ == '__main__':
    with torch.no_grad():
        main()
