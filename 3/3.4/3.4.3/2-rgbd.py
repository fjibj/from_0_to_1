#2．基于 RGB-D 摄像的方法
#以下是一个简化的基于 RGB-D 摄像的 3D 唇型检测代码示例，展示了如何使用深度信息来进行口型匹配。
import cv2
import numpy as np
import open3d as o3d

# 初始化深度摄像机
kinect = cv2.VideoCapture(cv2.CAP_OPENNI2)
if not kinect.isOpened() :
 raise Exception("Unable to open Kinect")

# 读取深度图像和RGB图像
ret, depth_frame = kinect.read()
ret, color_frame = kinect.read()

# 嘴唇区域提取（示例）
lip_region = color_frame[100 :200, 200 :400]

# 深度信息融合
depth_data = depth_frame[100 :200, 200 :400]
point_cloud = np.zeros((lip_region.shape[0], lip_region.shape[1], 3), dtype=np.float32)
for i in range(point_cloud.shape[0]) :
 for j in range(point_cloud.shape[1]) :
  depth = depth_data[i, j]
  if depth > 0 :
    point_cloud[i, j, 0] = j
    point_cloud[i, j, 1] = i
    point_cloud[i, j, 2] = depth

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# 三维形状建模
o3d.visualization.draw_geometries([pcd])

# 进行口型匹配预测
# ...

# 关闭深度摄像机
kinect.release()