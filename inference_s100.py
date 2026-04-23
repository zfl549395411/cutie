import numpy as np
from pathlib import Path
import os
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import cv2
import glob
import torch.nn.functional as F
from horizon_tc_ui.hb_runtime import HBRuntime
from cutie.inference.memory_manager import MemoryManager
from cutie.inference.object_manager import ObjectManager
from cutie.inference.image_feature_store import ImageFeatureStore
from cutie.utils.tensor_utils import pad_divide_by, unpad, aggregate
from cutie.utils.inference import *
from typing import List, Optional, Iterable, Dict
import logging
import cv2
import numpy as np
import torch.nn.functional as F

def process(mask, mask_id, mask_save_path):
    # mask = torch.argmax(prob, dim=0).cpu()
    out_mask = mask.cpu().numpy().astype(np.uint8)
    out_img = Image.fromarray(out_mask)
    # out_img.putpalette(davis_palette)
    os.makedirs(mask_save_path, exist_ok=True)
    out_img.save(os.path.join(mask_save_path, mask_id + '.png'))
    return mask

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

def get_sub_folders(root_path):
    sub_folders = [f for f in Path(root_path).iterdir() if f.is_dir()]
    sub_folder_names = [f.name for f in sub_folders]
    abosute_path = []
    for sub_folder in sub_folder_names:
        abosute_path.append(os.path.join(root_path, sub_folder))
    return abosute_path
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

log = logging.getLogger()

class Config:
    def __init__(self, ):
        self.mem_every = 4
        self.stagger_updates = 5
        self.chunk_size = -1
        self.save_aux = False
        self.max_internal_size = -1
        self.flip_aug = False
        self.sensory_dim = 256
        self.topk = 30
        self.use_long_term = False
        self.count_long_term_usage = False
        self.max_mem_frames = 4



class InferenceCore_s100:

    def __init__(self,
                 image_feature_store: ImageFeatureStore = None):
        self.cfg = Config()
        self.mem_every = self.cfg.mem_every
        stagger_updates = self.cfg.stagger_updates
        self.chunk_size = self.cfg.chunk_size # 一次性处理的帧数
        self.save_aux = self.cfg.save_aux # 是否保存辅助信息
        self.max_internal_size = self.cfg.max_internal_size
        self.flip_aug = self.cfg.flip_aug

        self.curr_ti = -1
        self.last_mem_ti = 0 # 上一次更新记忆的时间戳
        # at which time indices should we update the sensory memory
        # 交叉更新
        if stagger_updates >= self.mem_every:
            self.stagger_ti = set(range(1, self.mem_every + 1))
        else:
            self.stagger_ti = set(
                np.round(np.linspace(1, self.mem_every, stagger_updates)).astype(int))
        self.object_manager = ObjectManager()
        self.memory = MemoryManager(cfg=self.cfg, object_manager=self.object_manager) # 记忆管理器

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore()
        else:
            self.image_feature_store = image_feature_store # 图像特征存储器

        self.last_mask = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(cfg=self.cfg, object_manager=self.object_manager)

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_sensory_memory()

    def update_config(self, cfg):
        self.mem_every = cfg.mem_every
        self.memory.update_config(cfg)

    def _add_memory(self,
                    image: torch.Tensor,
                    pix_feat: torch.Tensor,
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True,
                    force_permanent: bool = False) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used.
        image: RGB image, (1/2)*3*H*W
        pix_feat: from the key encoder, (1/2)*_*H*W
        prob: (1/2)*num_objects*H*W, in [0, 1]
        key/shrinkage/selection: for anisotropic l2, (1/2)*_*H*W
        selection can be None if not using long-term memory
        is_deep_update: whether to use deep update (e.g. with the mask encoder)
        force_permanent: whether to force the memory to be permanent
        """
        if prob.shape[1] == 0:
            # nothing to add
            log.warn('Trying to add an empty object mask to memory!')
            return

        if force_permanent:
            as_permanent = 'all'
        else:
            as_permanent = 'first'

        # 每个目标有一个短期的记忆，初始化为0，1*256*H*W
        self.memory.initialize_sensory_if_needed(key, self.object_manager.all_obj_ids)

        sensory =  self.memory.get_sensory(self.object_manager.all_obj_ids).squeeze(0) # for 4 dim
        # prob_round = torch.round(prob)
        encode_mask_input = {'image':image, 'pix_feat':pix_feat, 'sensory':sensory, 'prob':prob}
        # breakpoint()
        # msk_value, sensory, obj_value = onnx_infer(encode_mask_input, model_name="encode_mask")
        msk_value, sensory, obj_value = hbm_infer(encode_mask_input, model_name="encode_mask")
        # save2txt(tensor=msk_value, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/msk_value", name=f"msk_value_{self.curr_ti}.txt")
        # save2txt(tensor=sensory, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/sensory", name=f"sensory_{self.curr_ti}.txt")
        # save2txt(tensor=obj_value, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/obj_value", name=f"obj_value_{self.curr_ti}.txt")
        # save2txt(tensor=image, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/image", name=f"image_{self.curr_ti}.txt")
        # save2txt(tensor=pix_feat, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/pix_feat", name=f"pix_feat_{self.curr_ti}.txt")
        # save2txt(tensor=sensory, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/sensory", name=f"sensory_{self.curr_ti}.txt")
        # save2txt(tensor=prob, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/encode_mask/prob", name=f"prob_{self.curr_ti}.txt")

        # msk_value, sensory, obj_value, _ = self.network.encode_mask(
        #     image,
        #     pix_feat,
        #     self.memory.get_sensory(self.object_manager.all_obj_ids),
        #     prob,
        #     deep_update=is_deep_update,
        #     chunk_size=self.chunk_size,
        #     need_weights=self.save_aux)
        self.memory.add_memory(key,
                               shrinkage,
                               msk_value,
                               obj_value,
                               self.object_manager.all_obj_ids,
                               selection=selection,
                               as_permanent=as_permanent)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids) # cfg.mem_every=4保存一次sensory

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 pix_feat: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        key/selection: for anisotropic l2: (1/2) * _ * H * W
        pix_feat: from the key encoder, (1/2) * _ * H * W
        ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        update_sensory: whether to update the sensory memory

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        bs = key.shape[0]
        if self.flip_aug:
            assert bs == 2
        else:
            assert bs == 1

        if not self.memory.engaged:
            log.warn('Trying to segment without any memory!')
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)
        # 根据相似性加权融合历史work_memory视觉特征并利用transformer将object_memory和object_memory深度融合
        # 每个目标都输出自己的memroy_readout，输出是1*256*H*W，其实就是聚合后的特征
        memory_readout = self.memory.read(pix_feat, key, selection, self.last_mask, current_ti=self.curr_ti)
        memory_readout = self.object_manager.realize_dict(memory_readout) # 单目标先不导出
        # 网络输出每个目标的sensory 1*num_object*256*H*W,加背景的概率1*num_object+1*H*W
        
        # sensory, _, pred_prob_with_bg = self.network.segment(ms_features,
        #                                                      memory_readout,
        #                                                      self.memory.get_sensory(
        #                                                          self.object_manager.all_obj_ids),
        #                                                      chunk_size=self.chunk_size,
        #                                                      update_sensory=update_sensory,
        #                                                      current_ti = self.curr_ti)
        # remove batch dim
        # if self.flip_aug:
        #     # average predictions of the non-flipped and flipped version
        #     pred_prob_with_bg = (pred_prob_with_bg[0] +
        #                          torch.flip(pred_prob_with_bg[1], dims=[-1])) / 2
        # else:
        #     pred_prob_with_bg = pred_prob_with_bg[0]
        readout_memory = memory_readout.squeeze(0)
        sensory_ = self.memory.get_sensory(self.object_manager.all_obj_ids).squeeze(0)
        segment_input = {'f8':ms_features[1], 'f4':ms_features[2], 'readout_memory':readout_memory, 'sensory': sensory_}
        # sensory, net_mask, pred_prob_with_bg, mask = onnx_infer(segment_input, model_name="segment")
        sensory, net_mask, pred_prob_with_bg, mask = hbm_infer(segment_input, model_name="segment")
        # save2txt(tensor=ms_features[1], save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/f8", name=f"f8_{self.curr_ti}.txt")
        # save2txt(tensor=ms_features[2], save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/f4", name=f"f4_{self.curr_ti}.txt")
        # save2txt(tensor=readout_memory, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/readout_memory", name=f"readout_memory_{self.curr_ti}.txt")
        # save2txt(tensor=sensory_, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/sensory_in", name=f"sensory_{self.curr_ti}.txt")
        # save2txt(tensor=sensory, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/sensory_out", name=f"sensory_{self.curr_ti}.txt")
        # save2txt(tensor=net_mask, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/net_mask", name=f"net_mask_{self.curr_ti}.txt")
        # save2txt(tensor=pred_prob_with_bg, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/pred_prob_with_bg", name=f"pred_prob_with_bg_{self.curr_ti}.txt")
        # save2txt(tensor=mask, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/segment/mask", name=f"mask_{self.curr_ti}.txt")

        # 保留最新的sensory_memory
        if update_sensory:
            self.memory.update_sensory(sensory, self.object_manager.all_obj_ids)
        pred_prob_with_bg = pred_prob_with_bg.squeeze(0)
        return pred_prob_with_bg, net_mask, mask

    def step(self,
             image: torch.Tensor,
             mask: Optional[torch.Tensor] = None,
             objects: Optional[List[int]] = None,
             *,
             idx_mask: bool = True,
             end: bool = False,
             delete_buffer: bool = True,
             force_permanent: bool = False) -> torch.Tensor:
        """
        Take a step with a new incoming image.
        If there is an incoming mask with new objects, we will memorize them.
        If there is no incoming mask, we will segment the image using the memory.
        In both cases, we will update the memory and return a segmentation.

        image: 3*H*W
        mask: H*W (if idx mask) or len(objects)*H*W or None
        objects: list of object ids that are valid in the mask Tensor.
                The ids themselves do not need to be consecutive/in order, but they need to be 
                in the same position in the list as the corresponding mask
                in the tensor in non-idx-mask mode.
                objects is ignored if the mask is None. 
                If idx_mask is False and objects is None, we sequentially infer the object ids.
        idx_mask: if True, mask is expected to contain an object id at every pixel.
                  If False, mask should have multiple channels with each channel representing one object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False 
        delete_buffer: whether to delete the image feature buffer after this step
        force_permanent: the memory recorded this frame will be added to the permanent memory
        """
        
        if objects is None and mask is not None:
            assert not idx_mask
            objects = list(range(1, mask.shape[0] + 1))

        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        # 翻转增强，如果启用翻转增强，则将输入图像镜像翻转两次
        if self.flip_aug:
            image = torch.cat([image, torch.flip(image, dims=[-1])], dim=0)

        # whether to update the working memory
        # 如果有mask强制更新，否则cfg.mem_every帧强制更新一次
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end) #初始帧的结果需要保存下来
        
        # segment when there is no input mask or when the input mask is incomplete
        # 如果无mask输入或者存在新的object时触发完整分割
        need_segment = (mask is None) or (self.object_manager.num_obj > 0
                                          and not self.object_manager.has_all(objects)) # 除了第一帧，每帧都分割
        # sensor memory更新节奏，与完整的memory(long memory和pixel memory)分开，避免冲高
        update_sensory = ((self.curr_ti - self.last_mem_ti) in self.stagger_ti) and (not end) # 除了第一帧，每帧都更新
        # if (self.curr_ti < 200):
        #     np.save(f"/media/sti/B20F0FD71CF7DE70/cutie/calib_data/image_encoder/image_encoder_image_{self.curr_ti}", image.clone().cpu().numpy())

        # encoding the image
        # 全量特征图 全局pix特征
        # 输入为1*3*H*W，输出图像编码的三种下采样特征ms_feat为1*(64*n)*(H/n)*(W/n),n为4 8 16，再对f16卷积通道下采样到指定256
        image_encoder_input = {'input_image':image}
        # f16, f8, f4, pix_feat, key, shrinkage, selection = onnx_infer(image_encoder_input, model_name="image_encoder")
        print(self.curr_ti)
        f16, f8, f4, pix_feat, key, shrinkage, selection = hbm_infer(image_encoder_input, model_name="image_encoder")
        ms_feat = [f16, f8, f4]
        # save2txt(tensor=f8, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/f8", name=f"f8_{self.curr_ti}.txt")
        # save2txt(tensor=f4, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/f4", name=f"f4_{self.curr_ti}.txt")
        # save2txt(tensor=pix_feat, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/pix_feat", name=f"pix_feat_{self.curr_ti}.txt")
        # save2txt(tensor=key, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/key", name=f"key_{self.curr_ti}.txt")
        # save2txt(tensor=shrinkage, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/shrinkage", name=f"shrinkage_{self.curr_ti}.txt")
        # save2txt(tensor=selection, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/selection", name=f"selection_{self.curr_ti}.txt")
        # save2txt(tensor=image, save_root="/open_explorer/moka/cutie_onnx_400_624/inference_debug/image_encoder/input_image", name=f"input_image_{self.curr_ti}.txt")

        # breakpoint()    
        # ms_feat, pix_feat, key, shrinkage, selection = self.image_feature_store.get_features(self.curr_ti, image) # ms_feat.shape=([1, 256, 20, 31],[1, 128, 40, 62],[1, 64, 80, 124]) pix_feat.shape=[1, 256, 20, 31]
        # 注意力机制相关特征 来自于f16用于相似性度量的key 用于削减注意力峰值的shrinkage 用于低响应mask的selection
        # key: 1*64*(H/16)*(W/16), shrinkage：拍平了 所以是1*1*(H/16)*(W/16)，selection：每个特征值的mask,1*64*(H/16)*(W/16)
        # key也是一个全局特征,所以注意力实际上是查询的需要关注的空间
        # key, shrinkage, selection = self.image_feature_store.get_key(self.curr_ti, image) # key.shape=[1, 64, 20, 31]  shrinkage.shape=[1, 1, 20, 31] selection.shape=[1, 64, 20, 31]  
       

        # segmentation from memory if needed
        if need_segment:
           # 第一帧不需要分割，因为传入了mask
            # pred_prob_with_bg = self._segment(key,
            #                                   selection,
            #                                   pix_feat,
            #                                   ms_feat,
            #                                   update_sensory=update_sensory)
            pred_prob_with_bg, net_mask, mask_ = self._segment(key,
                                              selection,
                                              pix_feat,
                                              ms_feat,
                                              update_sensory=update_sensory)

        # use the input mask if provided and then this if-else will not execute
        if mask is not None:
            # inform the manager of the new objects, and get a list of temporary id
            # temporary ids -- indicates the position of objects in the tensor
            # (starts with 1 due to the background channel)
            corresponding_tmp_ids, _ = self.object_manager.add_new_objects(objects)
            # 输入必须是16的整数倍
            mask, _ = pad_divide_by(mask, 16)
            if need_segment:
                # merge predicted mask with the incomplete input mask
                pred_prob_no_bg = pred_prob_with_bg[1:]
                # use the mutual exclusivity of segmentation
                if idx_mask:
                    pred_prob_no_bg[:, mask > 0] = 0
                else:
                    pred_prob_no_bg[:, mask.max(0) > 0.5] = 0

                new_masks = []
                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                    if idx_mask:
                        this_mask = (mask == objects[mask_id]).type_as(pred_prob_no_bg)
                    else:
                        this_mask = mask[tmp_id]
                    if tmp_id > pred_prob_no_bg.shape[0]:
                        new_masks.append(this_mask.unsqueeze(0))
                    else:
                        # +1 for padding the background channel
                        pred_prob_no_bg[tmp_id - 1] = this_mask
                # new_masks are always in the order of tmp_id
                mask = torch.cat([pred_prob_no_bg, *new_masks], dim=0)
            elif idx_mask:
                # objects = [255]
                # simply convert cls to one-hot representation
                if len(objects) == 0:
                    if delete_buffer:
                        self.image_feature_store.delete(self.curr_ti)
                    log.warn('Trying to insert an empty mask as memory!')
                    return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                                       device=key.device,
                                       dtype=key.dtype)
                # 输出为len(objects)*H*W的bool掩码
                mask = torch.stack(
                    [mask == objects[mask_id] for mask_id, _ in enumerate(corresponding_tmp_ids)],
                    dim=0)
    
            # 将所有通道不是
            # 通过这个之后增加一个通道，第一个通道是背景概率，其他通道是对应object的概率（所以需要temp_id，必须按顺序）
            pred_prob_with_bg = aggregate(mask, dim=0) # 初始化需要完成的操作
            pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0) # 初始化需要完成的操作
            # self.last_mask = mask.float().unsqueeze(0) # 端侧部署时可设置last_mask直接等于mask，不影响结果 , 这行为代码测试

        # 去除前景通道概率
        self.last_mask = pred_prob_with_bg[1:].unsqueeze(0) # 保存的是目标人物通道的mask

        # self.last_mask = pred_prob_with_bg[1:].unsqueeze(0) # 都是接近1的数和靠近0的数
        if self.flip_aug:
            self.last_mask = torch.cat(
                [self.last_mask, torch.flip(self.last_mask, dims=[-1])], dim=0)
        
        # save as memory if needed  
        if is_mem_frame or force_permanent:
            self._add_memory(image,
                             pix_feat,
                             self.last_mask,
                             key,
                             shrinkage,
                             selection,
                             force_permanent=force_permanent)
        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti) # delete_buffer 删除当前帧的图像
        output_prob = unpad(pred_prob_with_bg, self.pad)
    
        #----------------------------------------------------not important----------------------------------------------------#
        # if resize_needed:
        #     # restore output to the original size
        #     output_prob = F.interpolate(output_prob.unsqueeze(0),
        #                                 size=(h, w),
        #                                 mode='area',
        #                                 align_corners=False)[0]

        return output_prob    

    def delete_objects(self, objects: List[int]) -> None:
        """
        Delete the given objects from the memory.
        """
        self.object_manager.delete_objects(objects)
        self.memory.purge_except(self.object_manager.all_obj_ids)

    def output_prob_to_mask(self, output_prob: torch.Tensor) -> torch.Tensor:
        # mask = torch.round(output_prob[1])
        mask = torch.argmax(output_prob, dim=0) # [H, W]
        # index in tensor != object id -- remap the ids here
        new_mask = torch.zeros_like(mask)
        for tmp_id, obj in self.object_manager.tmp_id_to_obj.items():
            new_mask[mask == tmp_id] = obj.id
        return new_mask



@torch.inference_mode()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    # Typically, use one InferenceCore per video
    processor = InferenceCore_s100()
    # the processor matches the shorter edge of the input to this size
    # you might want to experiment with different sizes, -1 keeps the original size
    processor.max_internal_size = 480

    # ordering is important
    images = sorted(os.listdir(image_path))
    
    # mask for the first frame
    # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
    # and definitely NOT a colored RGB image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
    mask = Image.open(mask_path)
    if mask.mode not in ['L', 'P']:
    # 不是灰度图 → 转成灰度图
        mask = mask.convert('L')
    # breakpoint()

    # mask.save("/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track_1/masks/fixed_roi_mask_resize_224_384.png")
    assert mask.mode in ['L', 'P']

    # palette is for visualization
    palette = mask.getpalette()

    # the number of objects is determined by counting the unique values in the mask
    # common mistake: if the mask is resized w/ interpolation, there might be new unique values
    objects = np.unique(np.array(mask))
    # background "0" does not count as an object
    objects = objects[objects != 0].tolist()

    mask = torch.from_numpy(np.array(mask))

    for ti, image_name in enumerate(images):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))
        # image.save("/media/sti/B20F0FD71CF7DE70/FeishuDwonload/test_cutie/test_track_1/masks/image_resize_224_384.png")
        image = to_tensor(image).float() 
        image, mask = resize(image, mask, size=400) # image=[3,400,618]
        # breakpoint()
        if ti == 0:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory

            output_prob = processor.step(image) # [2, height, width]

        # convert output probabilities to an object mask

        mask = processor.output_prob_to_mask(output_prob)
        # # print(mask.shape)
        vedio_with_mask(image, mask, ti, save_path)




if __name__ == '__main__':
    # image_path = "/open_explorer/moka/cutie_onnx_s100/eval_data/subway/images"
    image_path = "/open_explorer/moka/cutie_onnx_s100/eval_data/test_track/images"
    # save_path = "/open_explorer/moka/cutie_onnx_400_624/vis/tset_read_memory_debug_clip_with_max_all_model"
    save_path = "/open_explorer/moka/cutie_onnx_400_624/vis/test_track_all_model_int16_hbm_400_624_no_padding"
    # mask_path = "/open_explorer/moka/cutie_onnx_s100/eval_data/subway/masks/fixed_roi_mask_50f.png"
    mask_path = "/open_explorer/moka/cutie_onnx_s100/eval_data/test_track/masks/fixed_roi_mask.png"
    # mask_path = "/open_explorer/moka/cutie_onnx_400_624/inference_debug/test_track_mask.png"
    os.makedirs(save_path, exist_ok=True)
    main()
