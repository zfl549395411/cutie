import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from cutie.model.utils.memory_utils import *
import torch.fx
import sys
image_width=480 # step函数中pad后的维度
image_height=304
def OnnxSimplify(model_path:str, output_path:str):
    from onnxsim import simplify
    import onnx
    device = torch.device("cpu")
    H=image_height//16
    W=image_width//16
    pix_feat = torch.randn(1, 256, H, W, device=device)         # 输入像素特征
    key = torch.randn(1, 64, H, W, device=device)                # key
    selection = torch.randn(1, 64, H, W, device=device)          # selection mask
    last_mask = torch.randn(1, 1, image_height, image_width, device=device)           # 上一帧的 mask

    work_len = H * W * 10
    long_len = 128 * 80
    mem_total_len = work_len + long_len

    mem_key = torch.randn(1, 64, mem_total_len, device=device)        # memory key
    mem_shrinkage = torch.randn(1, 1, mem_total_len, device=device)   # shrinkage
    mem_value = torch.randn(1, 256, mem_total_len, device=device)     # value

    sensory = torch.randn(1, 1, 256, H, W, device=device)              # sensory (全局引导)
    obj_mem = torch.randn(1, 1, 16, 257, device=device)                # object memory (历史)

    # 打包输入
    dummy_inputs = (
        pix_feat, key, selection, last_mask,
        mem_key, mem_shrinkage, mem_value,
        sensory, obj_mem
    )
    model = onnx.load(model_path)
    model_simplified, check = simplify(
        model,
        dynamic_input_shape=False,  
        skip_fuse_bn=False         
        # enable_shape_inference=True 
    )
    onnx.save(model_simplified, output_path)
    print(f"ONNX模型已简化并保存到：{output_path}")

# 一步推理
class FullTrackingWrapper(nn.Module):
    def __init__(self,  inferencecore):
        super().__init__()
        self.inferencecore = inferencecore

    def forward(self, image,f16,f8,f4,pix_feat,key,selection, last_mask,
                mem_key, mem_shrinkage, mem_value,
                sensory, obj_mem):
        """
        image:            1*3*h*w
        f16:          1*1024*H*W  H=h/16 W=w/16
        f8:           1*512*(h/8)*(w/8)
        f4:           1*256*(h/4)*(w/4)
        pix_feat:     1*256*H*W
        key:          1*64*H*W
        shrinkage:    1*1*H*W
        selection:    1*64*H*W
        last_mask:        1*1*h*w
        mem_key:          1*64*(H*W*n) + 1*64*(128*80)
        mem_shrinkage:    1*1*(H*W*n) + 1*1*(128*80)
        mem_value:        1*256*(H*W*n) + 1*256*(128*80)
        sensory:          1*1*256*H*W
        obj_mem:          1*1*16*257
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        # ============ Stage 2: memory 读取 ============
        similarity = get_similarity(mem_key, mem_shrinkage, key, selection, add_batch_dim=False)

        affinity, usage = do_softmax(similarity,
                                     top_k=self.inferencecore.memory.top_k,
                                     inplace=False,
                                     return_usage=True)

        visual_readout = self.inferencecore.memory._readout(affinity, mem_value)
        visual_readout=visual_readout.view(bs, 1, 256, h, w)

        pixel_readout = self.inferencecore.network.pixel_fusion(
            pix_feat, visual_readout, sensory, last_mask
        )

        obj_mem = obj_mem.unsqueeze(2)

        readout_memory, _ = self.inferencecore.network.readout_query(pixel_readout, obj_mem)

        # ============ Stage 3: segment ============
        ms_image_feat = [f16, f8, f4]
        sensory, _, pred_prob_with_bg = self.inferencecore.network.segment(
            ms_image_feat,
            readout_memory,
            sensory,
            chunk_size=-1,
            update_sensory=True
        )
        # 去掉batch获取带背景概率
        pred_prob_with_bg = pred_prob_with_bg[0]
        # 去除背景增加batch
        net_mask = pred_prob_with_bg[1:].unsqueeze(0)
        # 得到每个像素的分割mask
        mask = torch.argmax(pred_prob_with_bg, dim=0)

        # 找到每个像素的最大值（概率）
        max_prob = torch.max(pred_prob_with_bg, dim=0).values  # shape: [H, W]

        # 对低于阈值的区域将 mask 置为 0
        mask[max_prob < 0.1] = 0


        # ============ Stage 4: encode mask ============
        msk_value, sensory, obj_value, _ = self.inferencecore.network.encode_mask(
            image,
            pix_feat,
            sensory,
            net_mask,
            chunk_size=-1,
            need_weights=False
        )

        # ============ Final return ============
        return (
            msk_value, # 注意力机制四元素
            obj_value, # 目标级特征
            sensory,   # 图像感知特征
            mask,      # 像素类别mask
            net_mask,  # 概率
            usage,     # 注意力字典使用情况 
            pixel_readout,
            similarity,
            readout_memory,

        )

class ImageEncoderONNXWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        # image:        1*3*h*w 

        # 获取特征
        ms_features, pix_feat = self.network.encode_image(image)
        f16, f8, f4 = ms_features
        key, shrinkage, selection = self.network.transform_key(ms_features[0])
        # f16:          1*1024*H*W  H=h/16 W=w/16
        # f8:           1*512*(h/8)*(w/8)
        # f4:           1*256*(h/4)*(w/4)
        # pix_feat:     1*256*H*W
        # key:          1*64*H*W
        # shrinkage:    1*1*H*W
        # selection:    1*64*H*W

        return f16, f8, f4, pix_feat, key, shrinkage, selection
# 单目标读取融合记忆
class ReadMemoryONNXWrapper(nn.Module):
    def __init__(self, inferencecore):
        super().__init__()
        self.InferenceCore = inferencecore

    def forward(self, pix_feat, key, selection, last_mask,
                mem_key, mem_shrinkage, mem_value, sensory, obj_mem):
        # pix_feat:         1*256*H*W
        # key:              1*64*H*W
        # selection:        1*64*H*W
        # lask_mask:        1*1*h*w
        # mem_key:          worK:1*64*(H*W*10)+long:1*64*(128*80)
        # mem_shrinkage:    work:1*1*(H*W*10)+LONG:1*1*(128*80)
        # mem_value:        work:1*256*(H*W*10)+LONG:1*256*(128*80)
        # sensory:          1*1*256*H*W
        # obj_mem:          1*1*16*257


        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]

        # 计算相似度
        similarity = get_similarity(mem_key, mem_shrinkage, key, selection, add_batch_dim=False)

        # 注意力 softmax
        affinity, usage = do_softmax(similarity,
                                     top_k=self.InferenceCore.memory.top_k,
                                     inplace=False,
                                     return_usage=True)

        # 读取 memory 特征
        visual_readout = self.InferenceCore.memory._readout(
            affinity, mem_value) # torch.Size([1, 1, 256, 620])
        visual_readout=visual_readout.view(bs, 1, 256, h, w)
        
        # 融合像素特征
        pixel_readout = self.InferenceCore.network.pixel_fusion(
            pix_feat, visual_readout, sensory, last_mask
        )
        
        # 调整维度，确保 ONNX 支持
        obj_mem = obj_mem.unsqueeze(2)

        # 最终 memory 读取
        readout_memory, aux_features = self.InferenceCore.network.readout_query(
            pixel_readout, obj_mem
        )
        
        # readout:          1*1*256*H*W  
        # usage:            H*W*10+128*9
        # return pixel_readout
        return readout_memory, usage 

class SegMentONNXWrapper(nn.Module):
    def __init__(self, inferencecore):
        super().__init__()
        self.InferenceCore = inferencecore

    def forward(self, f16,f8,f4,readout_memory,sensory):
        # f16:              1*1024*H*W  H=h/16 W=w/16
        # f8:               1*512*(h/8)*(w/8)
        # f4:               1*256*(h/4)*(w/4)
        # readout:          1*1*256*H*W  
        # sensory:          1*1*256*H*W
        ms_image_feat=[f16,f8,f4]
        # sensory = sensory.unsqueeze(0) # for rk3588
        sensory, _, pred_prob_with_bg = self.InferenceCore.network.segment(ms_image_feat,readout_memory,sensory,chunk_size=-1,update_sensory=True)
        # sensory = sensory.squeeze(0) # for rk3588 

        # 去掉batch维度
        pred_prob_with_bg_ = pred_prob_with_bg[0]
        # 去除背景增加batch
        net_mask = pred_prob_with_bg_[1:].unsqueeze(0)
        # 取最大通道值用于分割
        mask = torch.argmax(pred_prob_with_bg_, dim=0, keepdim=True)
        # mask = mask.unsqueeze(0) # for 3588
        # mask = mask.unsqueeze(0).to(torch.int8) # int8 for rk3588
        # sensory:              1*1*256*H*W
        # net_mask:             1*1*h*w
        # pred_prob_with_bg:    2*h*w
        # mask:                 h*w
        return sensory,net_mask,pred_prob_with_bg,mask

class EncodeMaskWrapper(nn.Module):
    def __init__(self, inferencecore):
        super().__init__()
        self.InferenceCore = inferencecore
    def forward(self,image,pix_feat,sensory,prob):
        # image:            1*3*h*w 
        # pix_feat:         1*256*H*W
        # sensory:          1*1*256*H*W
        # prob:             1*1*h*w
        msk_value, sensory, obj_value, _ =self.InferenceCore.network.encode_mask(
            image,
            pix_feat,
            sensory,
            prob,
            chunk_size=-1,
            need_weights=False
        )
        return msk_value,sensory,obj_value
class CompressMemWrapper(nn.Module):
    def __init__(self, inferencecore):
        super().__init__()
        self.InferenceCore = inferencecore
    def forward(self,history_key,history_shrinkage,history_selection,hisotory_value,usage):
        # history_key:          1*64*(H*W*5)
        # history_shrinkage:    1*1*(H*W*5)
        # history_selection:    1*64*(H*W*5)
        # hisotory_value:       1*256*(H*W*5)
        # usage:                1*H*W*5
        # 压缩记忆
        input_value={1:hisotory_value}
        prototype_key, prototype_value,prototype_shrinkage = \
            self.InferenceCore.memory.consolidation(
                history_key, history_shrinkage, history_selection, input_value, usage
            )
        # prototype_key:     work:1*64*128
        # prototype_value:   work:1*256*128
        # prototype_shrinkage: work:1*1*128
        return prototype_key, prototype_value,prototype_shrinkage
        

def ExportFullStageOnnx(image_height,image_width):
    # 创建模型
    device = torch.device('cuda')

    # 创建模型并移动到 device
    cutie = get_default_model().to(device)
    for param in cutie.parameters():
        param.requires_grad = False
    cutie.eval()
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    wrapper = FullTrackingWrapper(processor)
    wrapper.eval()

    H=image_height//16 #224//16=14
    W=image_width//16 #384/16=24
    work_len = H * W * 10 #3360
    long_len = 128 * 80
    mem_total_len = work_len + long_len  #13360 

    image=torch.randn(1, 3, image_height,image_width,device=device)
    last_mask = torch.randn(1, 1, image_height, image_width, device=device)           # 上一帧的 mask
    mem_key = torch.randn(1, 64, mem_total_len, device=device)        # memory key
    mem_shrinkage = torch.randn(1, 1, mem_total_len, device=device)   # shrinkage
    mem_value = torch.randn(1, 256, mem_total_len, device=device)     # value
    sensory = torch.randn(1, 1, 256, H, W, device=device)              # sensory (全局引导)
    obj_mem = torch.randn(1, 1, 16, 257, device=device)                # object memory (历史)
    f16 = torch.randn(1, 1024, H, W).to(device)
    f8  = torch.randn(1, 512, H * 2, W * 2).to(device)
    f4  = torch.randn(1, 256,  H * 4, W * 4).to(device)
    pix_feat = torch.randn(1, 256, H, W, device=device)         # 输入像素特征
    key = torch.randn(1, 64, H, W, device=device)                # key
    selection = torch.randn(1, 64, H, W, device=device)          # selection mask

    dummy_inputs = (
        image,f16,f8,f4,pix_feat,key,selection, last_mask, mem_key, mem_shrinkage,
        mem_value, sensory, obj_mem
    )

    output_names = [
        "msk_value", "obj_value", "sensory_out",
        "mask","net_mask", "usage","pixel_readout",
            "similarity",
            "readout_memory",
    ]

    input_names = [
        "image","f16","f8","f4","pix_feat","key","selection", "last_mask",
        "mem_key", "mem_shrinkage", "mem_value",
        "sensory", "obj_mem"
    ]
    
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        "/media/sti/B20F0FD71CF7DE70/cutie/onnx_3588/full_tracking_debug.onnx",
        verbose=False,  # 开启详细日志
        do_constant_folding=True ,
        export_params=True,
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
    )

    
def ExportEncoderOnnx(image_height,image_width):
    # 创建模型
    device = torch.device('cuda')

    # 创建模型并移动到 device
    cutie = get_default_model().to(device)
    wrapper = ImageEncoderONNXWrapper(cutie)
    wrapper.eval()

    # 构造示例输入
    dummy_input = torch.randn(1, 3, image_height,image_width).to(device)

    # 导出 ONNX，指定输出命名
    torch.onnx.export(
        wrapper,
        dummy_input,
        "./onnx_3588/image_encoder.onnx",
        export_params=True,
        opset_version=13,
        input_names=['input_image'],
        output_names=['f16', 'f8', 'f4', 'pix_feat', 'key', 'shrinkage', 'selection'],
        dynamic_axes=None  # 如果你想支持动态尺寸可在此添加
    )

def ExportReadMemoryOnnx(image_height,image_width):
    device = torch.device('cuda')
    cutie = get_default_model().to(device)
    for param in cutie.parameters():
        param.requires_grad = False
    cutie.eval()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    wrapper=ReadMemoryONNXWrapper(processor)
    wrapper.eval()

     # 构造 dummy 输入
    H=image_height//16
    W=image_width//16
    pix_feat = torch.randn(1, 256, H, W, device=device)         # 输入像素特征
    key = torch.randn(1, 64, H, W, device=device)                # key
    selection = torch.randn(1, 64, H, W, device=device)          # selection mask
    last_mask = torch.randn(1, 1, image_height, image_width, device=device)           # 上一帧的 mask

    work_len = H * W * 10
    long_len = 128 * 20
    mem_total_len =  2480 #work_len + long_len

    mem_key = torch.randn(1, 64, mem_total_len, device=device)        # memory key
    mem_shrinkage = torch.randn(1, 1, mem_total_len, device=device)   # shrinkage
    mem_value = torch.randn(1, 256, mem_total_len, device=device)     # value

    sensory = torch.randn(1, 1, 256, H, W, device=device)              # sensory (全局引导)
    obj_mem = torch.randn(1, 1, 16, 257, device=device)                # object memory (历史)

    # 打包输入
    dummy_inputs = (
        pix_feat, key, selection, last_mask,
        mem_key, mem_shrinkage, mem_value,
        sensory, obj_mem
    )

    # 导出 ONNX
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        "./onnx_3588/read_memory.onnx",
        do_constant_folding=False ,
        export_params=True,
        opset_version=13,
        input_names=[
            "pix_feat", "key", "selection", "last_mask",
            "mem_key", "mem_shrinkage", "mem_value",
            "sensory", "obj_mem"
        ],
        
        output_names=["readout_memory", "usage"],
        dynamic_axes=None  # 如果你想支持动态尺寸可以进一步指定
    )


def ExportSegmentOnnx(image_height, image_width):
    device = torch.device("cuda")
    cutie = get_default_model().to(device)
    for param in cutie.parameters():
        param.requires_grad = False
    cutie.eval()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    wrapper = SegMentONNXWrapper(processor)
    wrapper.eval()

    H = image_height // 16
    W = image_width // 16
    if cutie.cfg.model.pixel_encoder.type == "resnet50":
        dummy_f16 = torch.randn(1, 1024, H, W).to(device)
        dummy_f8  = torch.randn(1, 512, H * 2, W * 2).to(device)
        dummy_f4  = torch.randn(1, 256,  H * 4, W * 4).to(device)
        dummy_readout = torch.randn(1, 1, 256, H, W).to(device)
        dummy_sensory = torch.randn(1, 1, 256, H, W).to(device)
        # dummy_readout = torch.randn(1, 256, H, W).to(device) # for rk3588
        # dummy_sensory = torch.randn(1, 256, H, W).to(device) # for rk3588
    elif cutie.cfg.model.pixel_encoder.type == "resnet18":
        print('yes')
        dummy_f16 = torch.randn(1, 256, H, W).to(device)
        dummy_f8  = torch.randn(1, 128, H * 2, W * 2).to(device)
        dummy_f4  = torch.randn(1, 64,  H * 4, W * 4).to(device)
        dummy_readout = torch.randn(1, 1, 256, H, W).to(device)
        dummy_sensory = torch.randn(1, 1, 256, H, W).to(device)
        # dummy_readout = torch.randn(1, 256, H, W).to(device) # for rk3588
        # dummy_sensory = torch.randn(1, 256, H, W).to(device) # for rk3588

    traced = torch.jit.trace(wrapper, (dummy_f16, dummy_f8, dummy_f4, dummy_readout, dummy_sensory))
    with open("traced_graph.txt", "w") as f:
        f.write(str(traced.graph))

    torch.onnx.export(
        wrapper,
        (dummy_f16, dummy_f8, dummy_f4, dummy_readout, dummy_sensory),
        "./onnx_3588/segment.onnx",
        input_names=["f16", "f8", "f4", "readout_memory", "sensory"],
        output_names=["sensory_out", "net_mask","pred_prob_with_bg","mask"],
        export_params=True,
        opset_version=13,  # 推荐 >=13，支持更多算子如 `einsum`
        do_constant_folding=True,
        verbose=False
    )

def ExportEncodeMaskOnnx(image_height, image_width):
    device = torch.device("cuda")
    cutie = get_default_model().to(device)
    for param in cutie.parameters():
        param.requires_grad = False
    cutie.eval()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    wrapper = EncodeMaskWrapper(processor).to(device)
    wrapper.eval()

    # 计算特征图尺寸（假设 backbone downsample = 16）
    H = image_height // 16
    W = image_width // 16

    # 准备静态 dummy 输入
    dummy_image = torch.randn(1, 3, image_height, image_width, device=device)
    dummy_pix_feat = torch.randn(1, 256, H, W, device=device)
    dummy_sensory = torch.randn(1, 1, 256, H, W, device=device)
    dummy_prob = torch.randn(1, 1, image_height, image_width, device=device)

    # 导出到 onnx
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_pix_feat, dummy_sensory, dummy_prob),
        "./onnx_3588/encode_mask.onnx",
        verbose=False,  # 开启详细日志
        input_names=["image", "pix_feat", "sensory", "prob"],
        output_names=["msk_value", "sensory_out", "obj_value"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=None  # 关键：固定 shape，禁用动态维度
    )

# ExportEncoderOnnx(image_height,image_width)

def ExportCompressMemOnnx(image_height, image_width):

    device = torch.device("cuda")
    
    # 加载模型并设置为 eval 模式
    cutie = get_default_model().to(device)
    for p in cutie.parameters():
        p.requires_grad = False
    cutie.eval()
    
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    wrapper = CompressMemWrapper(processor).to(device)
    wrapper.eval()

    # 固定历史帧数量为 5，则 HW5 = H * W * 5
    H = image_height // 16
    W = image_width // 16
    HW5 = H * W * 5

    # 构造 dummy inputs，注意尺寸一致
    dummy_history_key       = torch.randn(1, 64, HW5, device=device)
    dummy_history_shrinkage = torch.randn(1, 1, HW5, device=device)
    dummy_history_selection = torch.randn(1, 64, HW5, device=device)
    dummy_history_value     = torch.randn(1, 256, HW5, device=device)
    dummy_usage             = torch.randn(1, HW5, device=device)

    # 导出到 ONNX
    torch.onnx.export(
        wrapper,
        (dummy_history_key, dummy_history_shrinkage, dummy_history_selection, dummy_history_value, dummy_usage),
        "./onnx_3588/compress_mem.onnx",
        input_names=["history_key", "history_shrinkage", "history_selection", "history_value", "usage"],
        output_names=["prototype_key", "prototype_value", "prototype_shrinkage"],
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=None  # 固定尺寸
    )

# try:
# ExportEncoderOnnx(image_height, image_width)
# ExportEncodeMaskOnnx(image_height,image_width)
# ExportFullStageOnnx(image_height,image_width)
# ExportCompressMemOnnx(image_height, image_width)
# ExportReadMemoryOnnx(image_height,image_width)
# ExportSegmentOnnx(image_height, image_width)
# except Exception as e:
#     with open("error_log.txt", "w") as f:
#         import traceback
#         traceback.print_exc(file=f)
