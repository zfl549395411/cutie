from typing import List, Iterable
import torch
import torch.nn.functional as F


# STM
def pad_divide_by(in_img: torch.Tensor, d: int) -> (torch.Tensor, Iterable[int]):
    h, w = in_img.shape[-2:]

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array


def unpad(img: torch.Tensor, pad: Iterable[int]) -> torch.Tensor:
    if len(img.shape) == 4:
        if pad[2] + pad[3] > 0:
            img = img[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, pad[0]:-pad[1]]
    elif len(img.shape) == 3:
        if pad[2] + pad[3] > 0:
            img = img[:, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, pad[0]:-pad[1]]
    elif len(img.shape) == 5:
        if pad[2] + pad[3] > 0:
            img = img[:, :, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            img = img[:, :, :, :, pad[0]:-pad[1]]
    else:
        raise NotImplementedError
    return img

def prod(input, dim=1, keepdim=True, eps=1e-5):
    input_safe = input + eps
    log_input_safe = torch.log(input_safe).clamp(-12,0)
    sum_log = torch.sum(log_input_safe, dim=dim, keepdim=keepdim)
    prob_sim = torch.exp(sum_log)
    return prob_sim

# @torch.jit.script
def aggregate(prob: torch.Tensor, dim: int) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        prob = prob.float()
        new_prob = torch.cat([prod(1 - prob, dim=dim, keepdim=True), prob],
                             dim).clamp(1e-1, 1 - 1e-1) # prod operater is modified for rk3588 and s100 
        # # experiments verify that new_prob only need keep relative number relationship
        logits = torch.log(new_prob/(1-new_prob)) # 增强特征显著性

        #------------------------for quantization debug------------------------------
        # scale_np = 2*new_prob.max()/255
        # new_prob_f = torch.round(new_prob/scale_np).clamp(-128, 127)*scale_np
        
        # tmp_div = 1 / (1 - new_prob_f)
        # scale = 2*tmp_div.max()/255
        # tmp_div_f = torch.round(tmp_div/scale).clamp(-128, 127)*scale
        # tmp = new_prob * tmp_div_f # for debuging , (0.1111, 9)
        # scale_ = 2*tmp.max()/255
        # tmp_f = torch.round(tmp/scale_).clamp(-128, 127)*scale_
        # logits = torch.log(tmp_f)
   
        # # breakpoint()
        # # logits = torch.log((new_prob / (1 - new_prob))) # 增强特征显著性
        # # with open("/media/sti/B20F0FD71CF7DE70/cutie/debug_txt/log.txt", "w") as f:
        # #     for data in logits.flatten():
        # #         f.write(str(data))
        # #         f.write("\n")
        # # breakpoint()
        # return logits
        return logits

#------------------------------------origin code --------------------------------------
# # @torch.jit.script
# def aggregate(prob: torch.Tensor, dim: int) -> torch.Tensor:
#     with torch.cuda.amp.autocast(enabled=False):
#         prob = prob.float()
#         new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob],
#                              dim).clamp(1e-7, 1 - 1e-7)
#         logits = torch.log((new_prob / (1 - new_prob)))

#         return logits


# @torch.jit.script
def cls_to_one_hot(cls_gt: torch.Tensor, num_objects: int) -> torch.Tensor:
    # cls_gt: B*1*H*W
    B, _, H, W = cls_gt.shape
    one_hot = torch.zeros(B, num_objects + 1, H, W, device=cls_gt.device).scatter_(1, cls_gt, 1)
    return one_hot