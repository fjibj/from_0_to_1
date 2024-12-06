#1．基于单视角的方法（修改）
#以下是使用VIBE进行3D姿态估计的简化代码示例
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

video_file = 'sample_video.mp4'

image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

print(f'Input video number of frames {num_frames}')
orig_height, orig_width = img_shape[:2]

#多人跟踪
mot = MPT(
    device=device,
    batch_size=args.tracker_batch_size,
    display=args.display,
    detector_type=args.detector,
    output_format='dict',
    yolo_img_size=args.yolo_img_size,
)
tracking_results = mot(image_folder)

#定义VIBE模型
model = VIBE_Demo(
    seqlen=16,
    n_layers=2,
    hidden_size=1024,
    add_linear=True,
    use_residual=True,
).to(device)

#加载预训练模型权重
pretrained_file = download_ckpt(use_3dpw=False)
ckpt = torch.load(pretrained_file)
print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
ckpt = ckpt['gen_state_dict']
model.load_state_dict(ckpt, strict=False)
model.eval()
print(f'Loaded pretrained weights from \"{pretrained_file}\"')

#对每个人分别使用VIBE模型推理
print(f'Running VIBE on each tracklet...')
vibe_time = time.time()
vibe_results = {}
for person_id in tqdm(list(tracking_results.keys())):
    bboxes = tracking_results[person_id]['bbox']
    joints2d = tracking_results[person_id]['joints2d']

    frames = tracking_results[person_id]['frames']

    dataset = Inference(
        image_folder=image_folder,
        frames=frames,
        bboxes=bboxes,
        joints2d=joints2d,
        scale=bbox_scale,
    )

    bboxes = dataset.bboxes
    frames = dataset.frames
    has_keypoints = True if joints2d is not None else False

    dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

    with torch.no_grad():

        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

        for batch in dataloader:
            if has_keypoints:
                batch, nj2d = batch
                norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

            batch = batch.unsqueeze(0)
            batch = batch.to(device)

            batch_size, seqlen = batch.shape[:2]
            output = model(batch)[-1]

            pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
            smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)
        smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
        del batch

    #保存结果到.pkl文件
    pred_cam = pred_cam.cpu().numpy()
    pred_verts = pred_verts.cpu().numpy()
    pred_pose = pred_pose.cpu().numpy()
    pred_betas = pred_betas.cpu().numpy()
    pred_joints3d = pred_joints3d.cpu().numpy()
    smpl_joints2d = smpl_joints2d.cpu().numpy()

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bboxes,
        img_width=orig_width,
        img_height=orig_height
    )

    joints2d_img_coord = convert_crop_coords_to_orig_img(
        bbox=bboxes,
        keypoints=smpl_joints2d,
        crop_size=224,
    )

    output_dict = {
        'pred_cam': pred_cam,
        'orig_cam': orig_cam,
        'verts': pred_verts,
        'pose': pred_pose,
        'betas': pred_betas,
        'joints3d': pred_joints3d,
        'joints2d': joints2d,
        'joints2d_img_coord': joints2d_img_coord,
        'bboxes': bboxes,
        'frame_ids': frames,
    }

    vibe_results[person_id] = output_dict

del model

end = time.time()
fps = num_frames / (end - vibe_time)

print(f'VIBE FPS: {fps:.2f}')
total_time = time.time() - total_time
print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

#渲染结果到一个视频文件
renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

output_img_folder = f'{image_folder}_output'
os.makedirs(output_img_folder, exist_ok=True)

print(f'Rendering output video, writing frames to {output_img_folder}')

frame_results = prepare_rendering_results(vibe_results, num_frames)
mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

image_file_names = sorted([
    os.path.join(image_folder, x)
    for x in os.listdir(image_folder)
    if x.endswith('.png') or x.endswith('.jpg')
])

for frame_idx in tqdm(range(len(image_file_names))):
    img_fname = image_file_names[frame_idx]
    img = cv2.imread(img_fname)

    for person_id, person_data in frame_results[frame_idx].items():
        frame_verts = person_data['verts']
        frame_cam = person_data['cam']

        mc = mesh_color[person_id]

        mesh_filename = None

        img = renderer.render(
            img,
            frame_verts,
            cam=frame_cam,
            color=mc,
            mesh_filename=mesh_filename,
        )

    cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

#保存最终的视频到sample_video_vibe_result.mp4
vid_name = os.path.basename(video_file)
save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
save_name = os.path.join(output_path, save_name)
print(f'Saving result video to {save_name}')
images_to_video(img_folder=output_img_folder, output_vid_file=save_name)

#更多VIBE相关内容请参考https://github.com/mkocabas/VIBE