import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import cv2
import glob
import torch.nn.functional as F
def process(mask, mask_id, mask_save_path):
    # mask = torch.argmax(prob, dim=0).cpu()
    out_mask = mask.cpu().numpy().astype(np.uint8)
    out_img = Image.fromarray(out_mask)
    # out_img.putpalette(davis_palette)
    os.makedirs(mask_save_path, exist_ok=True)
    out_img.save(os.path.join(mask_save_path, mask_id + '.png'))
    return mask

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    # 获取视频原帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 可选：获取视频总帧数、宽高
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'fps={fps}, total_frames={total_frames}')
    # 释放资源
    cap.release()
    

def resize(image, mask, size=480):
            h, w = image.shape[-2:]
            min_side = min(h, w)
            new_h = int(h / min_side * size)
            new_w = int(w / min_side * size)
            image = F.interpolate(image.unsqueeze(0),
                                      size=(new_h, new_w),
                                      mode='bilinear',
                                      align_corners=False)[0]
                
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                             size=(new_h, new_w),
                                             mode='bilinear')[0, 0].round().long()
       
            return image, mask
                    
def frames_to_vedio(frame_dir, output_video_path, fps=25):
    """
    从帧目录合成视频
    :param frame_dir: 帧文件所在目录（如 "./frames"）
    :param output_video_path: 输出视频路径（如 "./output.mp4"）
    :param fps: 视频帧率
    """
    # 1. 获取所有帧文件路径，按名称排序（关键！避免帧顺序错乱）
    frame_paths = sorted(os.listdir(frame_dir))
    print(len(frame_paths))
    

    if not frame_paths:
        raise FileNotFoundError(f"未在 {frame_dir} 找到任何帧文件")

    # 2. 读取第一帧，获取帧尺寸（所有帧尺寸必须和第一帧一致）
    first_frame = cv2.imread(os.path.join(frame_dir,frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"第一帧 {frame_paths[0]} 加载失败")
    height, width = first_frame.shape[:2]
    print(f"帧尺寸：{height}×{width}，总帧数：{len(frame_paths)}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)  # VideoWriter要求尺寸为 (宽, 高)
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器，请检查编码器和输出路径")

    # 4. 循环写入所有帧
    for idx in range(len(frame_paths)):
        frame = cv2.imread(os.path.join(frame_dir,str(idx)+'.jpg'))
        # 校验当前帧尺寸是否和第一帧一致
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        video_writer.write(frame)
        # 打印进度（可选）
        if (idx + 1) % 50 == 0:
            print(f"已写入 {idx+1}/{len(frame_paths)} 帧")

    # 5. 释放资源（必须！否则视频无法播放）
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"视频合成完成！保存路径：{output_video_path}")

def vedio_with_mask(image, mask, image_id, save_path, color=(0, 0, 255), alpha=0.4):
    """
    :param image: PIL Image或PyTorch张量（C,H,W）
    :param mask: PyTorch张量（H,W），值0/255（uint8）
    :param image_id: 帧ID
    :param color: 蒙版BGR颜色
    :param alpha: 透明度
    """
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()  # C,H,W → H,W,C
        if image_np.dtype == np.float32 or image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image):
        image_np = np.array(image, dtype=np.uint8)  # H,W,C（RGB）
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # RGB→BGR
    else:
        raise ValueError(f"不支持的图像类型：{type(image)}，仅支持PIL Image/PyTorch张量")

    mask_np = mask.cpu().numpy()
    if len(mask_np.shape) == 3:
        mask_np = mask_np.transpose(1,2,0)[:, :, 0]  # 3,H,W → H,W,3 → 单通道
    
    mask_np = np.where(mask_np >= 128, 255, 0).astype(np.uint8)  # 严格二值化

    # 蒙版创建+融合 
    mask_color = np.zeros_like(image_np, dtype=np.uint8)
    mask_color[mask_np == 255] = color  # 前景填充颜色

    # 融合（输入均为NumPy数组，无类型错误）
    blended = cv2.addWeighted(image_np, 1 - alpha, mask_color, alpha, 0)

    cv2.imwrite(os.path.join(save_path, f"{image_id}.jpg"), blended)
    

@torch.inference_mode()
# @torch.cuda.amp.autocast()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model().cuda()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    # the processor matches the shorter edge of the input to this size
    # you might want to experiment with different sizes, -1 keeps the original size
    processor.max_internal_size = 320

    # ordering is important
    images = sorted(os.listdir(image_path))
    
    # mask for the first frame
    # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
    # and definitely NOT a colored RGB image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
    mask = Image.open(mask_path)
    # mask.save("/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track_1/masks/fixed_roi_mask_resize_224_384.png")
    assert mask.mode in ['L', 'P']

    # palette is for visualization
    palette = mask.getpalette()

    # the number of objects is determined by counting the unique values in the mask
    # common mistake: if the mask is resized w/ interpolation, there might be new unique values
    objects = np.unique(np.array(mask))
    # background "0" does not count as an object
    objects = objects[objects != 0].tolist()

    mask = torch.from_numpy(np.array(mask)).cuda()

    for ti, image_name in enumerate(images[50:]):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))
        # image.save("/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track_1/masks/image_resize_224_384.png")
        image = to_tensor(image).cuda().float()
        image, mask = resize(image, mask, size=304)
        
        
    
        if ti == 0:
            print('Yes')
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image)
        
        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob)
    
        vedio_with_mask(image, mask, ti, save_path)

        # # visualize prediction
        # mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        # mask.putpalette(palette)
        # mask.save(os.path.join("/media/sti/B20F0FD71CF7DE70/cutie/results_fa_lr_480_752", str(ti) + '.png'))
        # mask.show()  # or use mask.save(...) to save it somewhere



# image_path = '/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track/images' # test_track
# image_path = "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_longtime_disappear/images"
image_path = '/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/subway/images'
# mask_path = '/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track_1/masks/fixed_roi_mask.png' # test_track
# mask_path = "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_longtime_disappear/masks/fixed_roi_mask.png"
mask_path = '/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/subway/masks/fixed_roi_mask_50f.png'
# save_path = "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/vedio_with_mask/test_track_fa_lr_304_480_nr_nusl_max4f_res18" 
save_path = "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/vedio_with_mask/test_subway_fa_lr_304_480_nr_nusl_max4f_res18" 
# save_path = "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/vedio_with_mask/test_longtime_disappear_fa_lr_320_495_nr_nusl_max3f"
os.makedirs(save_path, exist_ok=True)
main()
# get_video_fps("/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/subway.mp4")
# frames_to_vedio(save_path, "/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/vedio_with_mask/vedio/test_track_fa_lr_304_480_nr_nusl_max4f_res18.mp4", fps=10)
