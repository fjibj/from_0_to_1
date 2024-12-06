#1．基于视频序列的表情跟踪
import cv2
# 读取视频文件
cap = cv2.VideoCapture('face_expression_video.mp4')
# 创建人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 创建光流法对象
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS + cv2.
TERM_CRITERIA_COUNT, 10, 0.03))
# 初始化特征点
old_frame = None
while cap.isOpened():
 ret, frame = cap.read()
 if not ret:
  break
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
 for (x, y, w, h) in faces:
  roi_gray = gray[y:y + h, x:x + w]
  p0 = cv2.goodFeaturesToTrack(roi_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
  # 计算光流
  p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, gray, p0, None, **lk_params)
  # 在图像上绘制光流轨迹
  for i, (new, old) in enumerate(zip(p1, p0)):
    if st[i] == 1:
      a, b = new.ravel()
      c, d = old.ravel()
      frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
 cv2.imshow("Face Expression Tracking", frame)
 k = cv2.waitKey(30) & 0xff
 if k == 27:
  break
cap.release()
cv2.destroyAllWindows()