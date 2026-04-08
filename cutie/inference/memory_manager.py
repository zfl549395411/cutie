import logging
from omegaconf import DictConfig
from typing import List, Dict
import torch
import numpy as np
from cutie.inference.object_manager import ObjectManager
from cutie.inference.kv_memory_store import KeyValueMemoryStore
from cutie.model.cutie import CUTIE
from cutie.model.utils.memory_utils import *

log = logging.getLogger()


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, cfg: DictConfig, object_manager: ObjectManager):
        self.object_manager = object_manager
        self.sensory_dim = cfg.model.sensory_dim
        self.top_k = cfg.top_k
        self.chunk_size = cfg.chunk_size

        self.save_aux = cfg.save_aux

        self.use_long_term = cfg.use_long_term
        # print(cfg.use_long_term)
        # breakpoint()
        self.count_long_term_usage = cfg.long_term.count_usage
        # subtract 1 because the first-frame is now counted as "permanent memory"
        # and is not counted towards max_mem_frames
        # but we want to keep the hyperparameters consistent as before for the same behavior
        if self.use_long_term:
            # 会比正常要长
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The sensory memory is stored as a dictionary indexed by object ids
        # each of shape bs * C^h * H * W
        self.sensory = {}

        # a dictionary indexed by object ids, each of shape bs * T * Q * C
        self.obj_v = {}

        self.work_mem = KeyValueMemoryStore(save_selection=self.use_long_term,
                                            save_usage=self.use_long_term)
        if self.use_long_term:
            self.long_mem = KeyValueMemoryStore(save_usage=self.count_long_term_usage)

        self.config_stale = True
        self.engaged = False

    def update_config(self, cfg: DictConfig) -> None:
        self.config_stale = True
        self.top_k = cfg['top_k']

        assert self.use_long_term == cfg.use_long_term, 'cannot update this'
        assert self.count_long_term_usage == cfg.long_term.count_usage, 'cannot update this'

        self.use_long_term = cfg.use_long_term
        self.count_long_term_usage = cfg.long_term.count_usage
        if self.use_long_term:
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

    def _readout(self, affinity, v) -> torch.Tensor:
        # affinity: bs*N*HW 
        # v: bs*C*N or bs*num_objects*C*N
        # returns bs*C*HW or bs*num_objects*C*HW
        if len(v.shape) == 3:
            
            # single object
            v = v.contiguous().cpu()
            affinity = affinity.contiguous().cpu()
            # bmm 要求输入为 3D 张量（B,M,K）×（B,K,N），和你的维度完全匹配
            result = torch.bmm(v, affinity)
            # return v @ affinity
            
            return result.cuda()
        else:
            bs, num_objects, C, N = v.shape # torch.Size([1, 1, 256, 620])
            v = v.view(bs, num_objects * C, N) # torch.Size([1, 256, 620])
            
            # out = torch.bmm(v, affinity)
            out = v @ affinity  # affinity 由620->1240->1860->2480 mem_max = 4 # out.shape=torch.Size([1, 256, 620])
            
            return out.view(bs, num_objects, C, -1)

    def _get_mask_by_ids(self, mask: torch.Tensor, obj_ids: List[int]) -> torch.Tensor:
        # -1 because the mask does not contain the background channel
        return mask[:, [self.object_manager.find_tmp_by_id(obj) - 1 for obj in obj_ids]]

    def _get_sensory_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=1)

    def _get_object_mem_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        if obj_ids[0] not in self.obj_v:
            # should only happen when the object transformer has been disabled
            return None
        return torch.stack([self.obj_v[obj] for obj in obj_ids], dim=1)

    def _get_visual_values_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        # All the values that the object ids refer to should have the same shape
        value = torch.stack([self.work_mem.value[obj] for obj in obj_ids], dim=1)
        if self.use_long_term and obj_ids[0] in self.long_mem.value:
            lt_value = torch.stack([self.long_mem.value[obj] for obj in obj_ids], dim=1)
            value = torch.cat([lt_value, value], dim=-1)

        return value

    def read(self, pix_feat: torch.Tensor, query_key: torch.Tensor, selection: torch.Tensor,
             last_mask: torch.Tensor, network: CUTIE, current_ti:int) -> Dict[int, torch.Tensor]:
        """
        Read from all memory stores and returns a single memory readout tensor for each object

        pix_feat: (1/2) x C x H x W
        query_key: (1/2) x C^k x H x W
        selection:  (1/2) x C^k x H x W
        last_mask: (1/2) x num_objects x H x W (at stride 16)
        return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        assert query_key.shape[0] == bs
        assert selection.shape[0] == bs
        assert last_mask.shape[0] == bs

        query_key = query_key.flatten(start_dim=2)  # bs*C^k*HW
        selection = selection.flatten(start_dim=2)  # bs*C^k*HW

        """
        Compute affinity and perform readout
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets #[0, [255]]
        for bucket_id, bucket in buckets.items():
            if self.use_long_term and self.long_mem.engaged(bucket_id):
                # Use long-term memory
                # 使用长时记忆并且拥有长时记忆就将长时记忆和工作记忆中的信息拼接拼接
                long_mem_size = self.long_mem.size(bucket_id)
                memory_key = torch.cat([self.long_mem.key[bucket_id], self.work_mem.key[bucket_id]],
                                       -1)
                shrinkage = torch.cat(
                    [self.long_mem.shrinkage[bucket_id], self.work_mem.shrinkage[bucket_id]], -1)
                # 相似性度量并获取topk的注意力索引和总权重
                # memory中每一帧的记忆都是一个B*1*H*W的key_value映射，即一个slot，一共有N帧所以有B*N*H*W，从这H个slot中选出topk个和当前目标query_key最相似的以及每个slot总的注意力权重
                
                similarity = get_similarity(memory_key, shrinkage, query_key, selection)
                affinity, usage = do_softmax(similarity,
                                             top_k=self.top_k,
                                             inplace=True,
                                             return_usage=True)
                """
                Record memory usage for working and long-term memory
                """
                # ignore the index return for long-term memory
                # 只记录work_memory的使用情况
                work_usage = usage[:, long_mem_size:]
                self.work_mem.update_bucket_usage(bucket_id, work_usage)

                if self.count_long_term_usage:
                    # ignore the index return for working memory
                    long_usage = usage[:, :long_mem_size]
                    self.long_mem.update_bucket_usage(bucket_id, long_usage)
            else:
                # no long-term memory
                memory_key = self.work_mem.key[bucket_id] # [1, 64, 2280]
                shrinkage = self.work_mem.shrinkage[bucket_id] # [1, 1, 2280]
                # 输出的是历史中每个记忆空间和当前的相似度相似度即1*1620*memory_frame*1620
                # if(13<=current_ti<214):
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/mem_key/read_memory_mem_key_{current_ti}.npy', memory_key.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/mem_shrinkage/read_memory_mem_shrinkage_{current_ti}.npy', shrinkage.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/key/read_memory_key_{current_ti}.npy', query_key.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/selection/read_memory_selection_{current_ti}.npy', selection.clone().cpu().numpy())
                
               
                similarity = get_similarity(memory_key, shrinkage, query_key, selection)

                if self.use_long_term:
                    affinity, usage = do_softmax(similarity,
                                                 top_k=self.top_k,
                                                 inplace=False,
                                                 return_usage=True)
                    self.work_mem.update_bucket_usage(bucket_id, usage)
                else:
                    affinity, usage = do_softmax(similarity, top_k=self.top_k, inplace=False)
            
           
            # 分片是为了加速并行
            if self.chunk_size < 1:
                object_chunks = [bucket] # [[255]]
            else:
                object_chunks = [
                    bucket[i:i + self.chunk_size] for i in range(0, len(bucket), self.chunk_size)
                ]
            
            # 融合sensormemory,历史mask，work_memory视觉特征（slot中的value），object_memory等，并通过transformer进行融合
            # 多目标同时进行
            for objects in object_chunks:
                this_sensory = self._get_sensory_by_ids(objects)
                # print(f'sensory={this_sensory.min()}  ')
                # 获取当前目标历史mask
                this_last_mask = self._get_mask_by_ids(last_mask, objects)
                # print(f'last_mask={this_last_mask.min()}  ')
                this_msk_value = self._get_visual_values_by_ids(objects)  # (1/2)*num_objects*C*N
                # 利用注意力机制从所有存储中对value进行加权计算，输出1*2*256*特征图宽高 msk_value和拼接次数有关，但这里输出无关

                # if(13<=current_ti<214):
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/mem_value/read_memory_mem_value_{current_ti}.npy', this_msk_value.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/pix_feat/read_memory_pix_feat_{current_ti}.npy', pix_feat.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/sensory/read_memory_sensory_{current_ti}.npy', this_sensory.clone().cpu().numpy())
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/last_mask/read_memory_last_mask_{current_ti}.npy', this_last_mask.clone().cpu().numpy())
                visual_readout = self._readout(affinity,
                                               this_msk_value).view(bs, len(objects), self.CV, h, w)
                # print(f'visual_readout={visual_readout.min()}  ')
                # 聚合信息尺度不变
                pixel_readout = network.pixel_fusion(pix_feat, visual_readout, this_sensory,
                                                     this_last_mask)
                # print(f'pixel_readout={pixel_readout.min()}  ')
                this_obj_mem = self._get_object_mem_by_ids(objects)
                this_obj_mem = this_obj_mem / this_obj_mem.abs().max() # this code line is added for s100 quantization 
                # print(f'this_obj_mem={this_obj_mem.min()}  ')
                this_obj_mem = this_obj_mem.unsqueeze(2) if this_obj_mem is not None else None

                # if(13<=current_ti<214):
                #     np.save(f'/media/sti/B20F0FD71CF7DE70/cutie/calib_data/read_memory/obj_mem/read_memory_obj_mem_{current_ti}.npy', this_obj_mem.clone().cpu().numpy())
                readout_memory, aux_features = network.readout_query(pixel_readout, this_obj_mem)
                # print(f'readout_memory={readout_memory.min()}\n')
                for i, obj in enumerate(objects):
                    all_readout_mem[obj] = readout_memory[:, i]

                if self.save_aux:
                    aux_output = {
                        'sensory': this_sensory,
                        'pixel_readout': pixel_readout,
                        'q_logits': aux_features['logits'] if aux_features else None,
                        'q_weights': aux_features['q_weights'] if aux_features else None,
                        'p_weights': aux_features['p_weights'] if aux_features else None,
                        'attn_mask': aux_features['attn_mask'].float() if aux_features else None,
                    }
                    self.aux = aux_output

        return all_readout_mem

    def add_memory(self,
                   key: torch.Tensor,
                   shrinkage: torch.Tensor,
                   msk_value: torch.Tensor,
                   obj_value: torch.Tensor,
                   objects: List[int],
                   selection: torch.Tensor = None,
                   *,
                   as_permanent: bool = False) -> None:
        # key: (1/2)*C*H*W
        # msk_value: (1/2)*num_objects*C*H*W
        # obj_value: (1/2)*num_objects*Q*C
        # objects contains a list of object ids corresponding to the objects in msk_value/obj_value
        bs = key.shape[0]
        assert shrinkage.shape[0] == bs
        assert msk_value.shape[0] == bs
        assert obj_value is None or obj_value.shape[0] == bs

        self.engaged = True
        if self.H is None or self.config_stale:
            self.config_stale = False
            # 特征图的宽高，第一次更新
            self.H, self.W = msk_value.shape[-2:]
            self.HW = self.H * self.W
            # convert from num. frames to num. tokens
            # 最多有4帧memory,每一个slot都存储了整个空间这一纬度的信息，所以最多有4*H*W
            self.max_work_tokens = self.max_mem_frames * self.HW
            if self.use_long_term:
                self.min_work_tokens = self.min_mem_frames * self.HW

        # key:   bs*C*N
        # value: bs*num_objects*C*N
        # 将其空间纬度拉平
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        # key的特征通道数
        self.CK = key.shape[1]

        # 每个目标空间纬度拍平
        msk_value = msk_value.flatten(start_dim=3)
        # value的特征通道数
        self.CV = msk_value.shape[2]

        if selection is not None:
            # not used in non-long-term mode
            selection = selection.flatten(start_dim=2)

        # insert object values into object memory
        if obj_value is not None:
            for obj_id, obj in enumerate(objects):
                if obj in self.obj_v:
                    """streaming average
                    each self.obj_v[obj] is (1/2)*num_summaries*(embed_dim+1)
                    first embed_dim keeps track of the sum of embeddings
                    the last dim keeps the total count
                    averaging in done inside the object transformer

                    incoming obj_value is (1/2)*num_objects*num_summaries*(embed_dim+1)
                    self.obj_v[obj] = torch.cat([self.obj_v[obj], obj_value[:, obj_id]], dim=0)
                    """
                    # 最后一维累加，其他维累加obj_value
                    last_acc = self.obj_v[obj][:, :, -1]
                    new_acc = last_acc + obj_value[:, obj_id, :, -1]

                    self.obj_v[obj][:, :, :-1] = (self.obj_v[obj][:, :, :-1] +
                                                  obj_value[:, obj_id, :, :-1])
                    self.obj_v[obj][:, :, -1] = new_acc
                else:
                    self.obj_v[obj] = obj_value[:, obj_id]
                

        # convert mask value tensor into a dict for insertion
        msk_values = {obj: msk_value[:, obj_id] for obj_id, obj in enumerate(objects)}
        self.work_mem.add(key,
                          msk_values,
                          shrinkage,
                          selection=selection,
                          as_permanent=as_permanent)

        for bucket_id in self.work_mem.buckets.keys():
            # long-term memory cleanup
            if self.use_long_term:
                # Do memory compressed if needed
                if self.work_mem.non_perm_size(bucket_id) >= self.max_work_tokens:
                    # Remove obsolete features if needed
                    if self.long_mem.non_perm_size(bucket_id) >= (self.max_long_tokens -
                                                                  self.num_prototypes):
                        self.long_mem.remove_obsolete_features(
                            bucket_id,
                            self.max_long_tokens - self.num_prototypes - self.buffer_tokens)

                    self.compress_features(bucket_id)
            else:
                # FIFO
                self.work_mem.remove_old_memory(bucket_id, self.max_work_tokens)

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        # purge certain objects from the memory except the one listed
        self.work_mem.purge_except(obj_keep_idx)
        if self.use_long_term and self.long_mem.engaged():
            self.long_mem.purge_except(obj_keep_idx)
        self.sensory = {k: v for k, v in self.sensory.items() if k in obj_keep_idx}

        if not self.work_mem.engaged():
            # everything is removed!
            self.engaged = False

    def compress_features(self, bucket_id: int) -> None:
        HW = self.HW

        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(bucket_id, 0, -self.min_work_tokens))

        # remove consolidated working memory
        # 删除以后剩下min_work_tokes+1680
        self.work_mem.sieve_by_range(bucket_id,
                                     0,
                                     -self.min_work_tokens,
                                     min_size=self.min_work_tokens)

        # add to long-term memory
        self.long_mem.add(prototype_key,
                          prototype_value,
                          prototype_shrinkage,
                          selection=None,
                          supposed_bucket_id=bucket_id)

    def consolidation(self, candidate_key: torch.Tensor, candidate_shrinkage: torch.Tensor,
                      candidate_selection: torch.Tensor, candidate_value: Dict[int, torch.Tensor],
                      usage: torch.Tensor) -> (torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # find the indices with max usage
        bs = candidate_key.shape[0]
        assert bs in [1, 2]

        prototype_key = []
        prototype_selection = []
        for bi in range(bs):
            _, max_usage_indices = torch.topk(usage[bi], k=self.num_prototypes, dim=-1, sorted=True)
            prototype_indices = max_usage_indices.flatten()
            prototype_key.append(candidate_key[bi, :, prototype_indices])
            prototype_selection.append(candidate_selection[bi, :, prototype_indices])
        prototype_key = torch.stack(prototype_key, dim=0)
        prototype_selection = torch.stack(prototype_selection, dim=0)
        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key,
                                    prototype_selection)
        affinity = do_softmax(similarity)

        # readout the values
        prototype_value = {k: self._readout(affinity, v) for k, v in candidate_value.items()}

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity, candidate_shrinkage)

        return prototype_key, prototype_value, prototype_shrinkage

    def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: List[int]):
        for obj in ids:
            if obj not in self.sensory:
                # also initializes the sensory memory
                bs, _, h, w = sample_key.shape
                self.sensory[obj] = torch.zeros((bs, self.sensory_dim, h, w),
                                                device=sample_key.device)

    def update_sensory(self, sensory: torch.Tensor, ids: List[int]):
        # sensory: 1*num_objects*C*H*W
        for obj_id, obj in enumerate(ids):
            self.sensory[obj] = sensory[:, obj_id]

    def get_sensory(self, ids: List[int]):
        # returns (1/2)*num_objects*C*H*W
        return self._get_sensory_by_ids(ids)

    def clear_non_permanent_memory(self):
        self.work_mem.clear_non_permanent_memory()
        if self.use_long_term:
            self.long_mem.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.sensory = {}
