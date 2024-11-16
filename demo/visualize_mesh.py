import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh

# 可视化3D网格函数
def vis_mesh(left_obj, right_obj, output_file):
    # 提取左手和右手的顶点和面
    left_verts, left_faces = np.array(left_obj.vertices), np.array(left_obj.faces)
    right_verts, right_faces = np.array(right_obj.vertices), np.array(right_obj.faces)
    
    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制左手 (红色)
    left_mesh = Poly3DCollection(left_verts[left_faces], facecolors='red', linewidths=0.1, edgecolors='r', alpha=0.5)
    ax.add_collection3d(left_mesh)
    
    # 绘制右手 (蓝色)
    right_mesh = Poly3DCollection(right_verts[right_faces], facecolors='blue', linewidths=0.1, edgecolors='b', alpha=0.5)
    ax.add_collection3d(right_mesh)
    
    # 设置坐标轴范围
    all_verts = np.vstack([left_verts, right_verts])  # 合并左手和右手顶点以设置范围
    ax.set_xlim([all_verts[:, 0].min(), all_verts[:, 0].max()])
    ax.set_ylim([all_verts[:, 1].min(), all_verts[:, 1].max()])
    ax.set_zlim([all_verts[:, 2].min(), all_verts[:, 2].max()])
    
    ax.view_init(elev=270, azim=90)  # 设置观察角度
    ax.axis('off')  # 关闭坐标轴
    
    # 保存 3D 网格图像
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    fig.canvas.draw()
    
    # 提取图像数据（用于进一步处理）
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # 关闭图形以释放内存
    return image

# 主函数
def main(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1, 21):
        
        original_img_path = os.path.join(input_folder, f'{i:02}.jpg')
        
        left_obj_path = os.path.join(input_folder, f'{i:02}_left.obj')
        right_obj_path = os.path.join(input_folder, f'{i:02}_right.obj')
        
        mesh_img = vis_mesh(trimesh.load(left_obj_path, file_type="obj"), 
                            trimesh.load(right_obj_path, file_type="obj"), 
                            os.path.join(output_folder, f'{i:02}_mesh.png'))
        
        img_resized = cv2.resize(cv2.imread(original_img_path), (mesh_img.shape[1], mesh_img.shape[0]))
        
        mesh_img = mesh_img[10:490, 50:650]
        mesh_img = cv2.resize(mesh_img, (img_resized.shape[1], img_resized.shape[0]))
        
        M = np.float32([[1, 0, 50], [0, 1, -30]])
        mesh_image_resized = cv2.warpAffine(mesh_img, M, (mesh_img.shape[1], mesh_img.shape[0]))
        mesh_image_resized[(mesh_image_resized[:, :, 0] == 255) & (mesh_image_resized[:, :, 1] == 255) & (mesh_image_resized[:, :, 2] == 255)] = [0, 0, 0]
        
        output_file = os.path.join(output_folder, f'{i:02}_resize.png')
        overlapped_image = cv2.addWeighted(img_resized, 0.5, mesh_image_resized, 1 - 0.5, 0)
        
        output_file = os.path.join(output_folder, f'{i:02}_overlapped.png')
        cv2.imwrite(output_file, overlapped_image)

if __name__ == '__main__':

    main('./imgs', './output')
