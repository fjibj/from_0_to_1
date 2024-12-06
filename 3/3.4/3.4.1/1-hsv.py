#1．基于颜色空间的方法
#以下是一个基于HSV颜色模型的简单代码示例，用于实现2D唇型检测。
import cv2
import numpy as np

def detect_lips_hsv(image) :
 # 将图像转换为HSV颜色空间
 hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

 # 定义唇部颜色范围
 lower_bound = np.array([H_min, S_min, V_min])
 upper_bound = np.array([H_max, S_max, V_max])

 # 根据颜色范围进行掩码操作
 mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

 # 执行形态学操作来增强唇部区域
 kernel = np.ones((5, 5), np.uint8)
 mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

 # 在原始图像上应用掩码
 result = cv2.bitwise_and(image, image, mask=mask)
 return result

# 调用函数进行唇型检测
image = cv2.imread('lip_image.jpg')
result_image = detect_lips_hsv(image)

# 显示结果图像
cv2.imshow('Lip Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()