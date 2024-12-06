#2．基于特征点的表情跟踪(修改)
#下面是一个简化的Python代码示例，展示了使用Menpo库实现AAM表情跟踪的方法。
from menpofit.aam import PatchAAM
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpodetect import load_dlib_frontal_face_detector
import menpo.io as mio
import matplotlib.pyplot as plt

#训练PatchAAM模型
patch_aam = PatchAAM(<training_images>, group='PTS', patch_shape=[(15, 15), (23, 23)],
                     diagonal=150, scales=(0.5, 1.0),
                     max_shape_components=20, max_appearance_components=150,
                     verbose=True)

#Lucas-Kanade推理器
fitter = LucasKanadeAAMFitter(patch_aam, lk_algorithm_cls=WibergInverseCompositional,
                     n_shape=[5, 20], n_appearance=[30, 150])
print(fitter)

#==如果不想自己训练模型，也可以使用预训练模型
from menpofit.aam.pretrained import load_balanced_frontal_face_fitter

fitter = load_balanced_frontal_face_fitter()
#==使用预训练模型结束

#加载人脸检测器
detect = load_dlib_frontal_face_detector()

#加载原图像并转化成灰度图像
image = mio.import_image('<要跟踪的图像路径>')
image = image.as_greyscale()

#脸部检测
bboxes = detect(image)

#裁剪图像
image = image.crop_to_landmarks_proportion(0.3, group='dlib_0')
bboxes[0] = image.landmarks['dlib_0'].lms

if len(bboxes) > 0:
    #推理
    result = fitter.fit_from_bb(image, bboxes[0], max_iters=[15, 5],
                    gt_shape=image.landmarks['PTS'].lms)
    print(result)

    #结果展示
    plt.subplot(131);
    image.view()
    bboxes[0].view(line_width=3, render_markers=False)
    plt.gca().set_title('Bounding box')

    plt.subplot(132)
    image.view()
    result.initial_shape.view(marker_size=4)
    plt.gca().set_title('Initial shape')

    plt.subplot(133)
    image.view()
    result.final_shape.view(marker_size=4, figure_size=(15, 13))
    plt.gca().set_title('Final shape')

#更多内容请参考Menpo-AAM.ipynb