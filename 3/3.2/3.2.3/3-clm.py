#3．基于运动模型的表情跟踪（修改）
#下面是CLM算法的Python代码示例。

#使用OpenFace命令行对视频中的人脸做表情跟踪，OpenFace内置了面部标志检测器和跟踪模型被卷积专家、约束局部模型 (CE-CLM)模型
!./OpenFace/build/bin/FaceLandmarkVidMulti -f video.mp4 -out_dir processed

#将视频转换成mp4格式
!ffmpeg -y -loglevel info -i processed/video.avi output.mp4

#显示结果
def show_local_mp4_video(file_name, width=640, height=480):
  import io
  import base64
  from IPython.display import HTML
  video_encoded = base64.b64encode(io.open(file_name, 'rb').read())
  return HTML(data='''<video width="{0}" height="{1}" alt="test" controls>
                        <source src="data:video/mp4;base64,{2}" type="video/mp4" />
                      </video>'''.format(width, height, video_encoded.decode('ascii')))

show_local_mp4_video('output.mp4', width=960, height=720)

#获取相关数据
import pandas as pd, seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt

df = pd.read_csv('processed/video.csv')
print(f"Max number of frames {df.frame.max()}", f"\nTotal shape of dataframe {df.shape}")
df.head()

#视频中有几张脸
print("Number of unique faces: ", len(df.face_id.unique()), "\nList of face_id's: ", df.face_id.unique())

#更多数据分析请参看OpenFace_Shared.ipynb