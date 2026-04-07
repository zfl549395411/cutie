import math
import torch
from typing import Optional, Union, Tuple


# @torch.jit.script
def get_similarity(mk: torch.Tensor,
                   ms: torch.Tensor,
                   qk: torch.Tensor,
                   qe: torch.Tensor,
                   add_batch_dim: bool = False) -> torch.Tensor:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened    
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)

    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See XMem's appendix for derivation
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq + two_ab)
    # similiraty data range (-380, -1)
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)  # B*N*HW CK=64
        similarity = similarity.clamp(-10,10)
        # ---------------------------------for debug---------------------------
        # simi = similarity.clone().flatten()
        # print(simi.min(),simi.max())
        # with open("./debug_txt/similiraty.txt", "w") as f:
        #     for num in simi:
        #         f.write(f"{num}\n")
        # breakpoint()


    else:
        similarity = similarity / math.sqrt(CK)  # B*N*HW
    # similiraty data range (-80, -1)

    #---------------------origin code, but not useful---------------------------
    # mask = (mk.abs().sum(dim=2) == 0)  # B x N, True 表示 padding（全0）的位置
    # mask = mask.unsqueeze(2)            # B x N x 1，方便广播
    # if(mask.any()):
    #         breakpoint()
    # similarity = similarity.masked_fill(mask, float('-1e4'))
    
    return similarity


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        # scale_similiraty = 2*similarity.abs().max()/255
        # similarity_f = torch.round(similarity/scale_similiraty).clamp(-128,127)*scale_similiraty
        values, indices = torch.topk(similarity, k=top_k, dim=1)
        # breakpoint()
        # affinity = similarity.exp_()
        # affinity /= torch.sum(affinity, dim=1, keepdim=True) 
        
        # 替代上述两行代码
        # affinity = torch.softmax(similarity, dim=1)

        
        x_exp = values.exp_().clamp(1e-2, 1)
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        # scale = 2*x_exp.abs().max()/255
        # x_exp_f = torch.round(x_exp/scale).clamp(-128,127)*scale
        # x_exp = x_exp_f
        # breakpoint()
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity)
            batch_idx = torch.arange(affinity.shape[0], device=affinity.device)[:, None, None]
            d2_idx = torch.arange(affinity.shape[2], device=affinity.device)[None, None, :]
            affinity[batch_idx, indices, d2_idx] = x_exp
            affinity = affinity.contiguous()
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


def get_affinity(mk: torch.Tensor, ms: torch.Tensor, qk: torch.Tensor,
                 qe: torch.Tensor) -> torch.Tensor:
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem
