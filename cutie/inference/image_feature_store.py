import warnings
from typing import Iterable
import torch

class ImageFeatureStore:
    """
    A cache for image features.
    These features might be reused at different parts of the inference pipeline.
    This class provide an interface for reusing these features.
    It is the user's responsibility to delete redundant features.

    Feature of a frame should be associated with a unique index -- typically the frame id.
    """
    def __init__(self, no_warning: bool = False):
        self._store = {}
        self.no_warning = no_warning

    # def _encode_feature(self, index: int, image: torch.Tensor) -> None:
    #     # 利用resnet进行一次前向推理，得到f4 f8 f16三种不同尺度的特征，并将f16特征降维后作为pix_feat
    #     ms_features, pix_feat = self.network.encode_image(image)
    #     key, shrinkage, selection = self.network.transform_key(ms_features[0])
    #     # 存储f16 f8 f4全量特征 f16投影特征 进一步3*3卷积后用于相似性度量的key 用于削减注意力峰值的shrinkage 用于低响应mask的selection
    #     self._store[index] = (ms_features, pix_feat, key, shrinkage, selection)

    # def get_features(self, index: int,
    #                  image: torch.Tensor) -> (Iterable[torch.Tensor], torch.Tensor):
    #     if index not in self._store:
    #         self._encode_feature(index, image)

    #     return self._store[index]

    # def get_key(self, index: int,
    #             image: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    #     if index not in self._store:
    #         self._encode_feature(index, image)

    #     return self._store[index][2:]

    def delete(self, index: int) -> None:
        if index in self._store:
            del self._store[index]

    def __len__(self):
        return len(self._store)

    def __del__(self):
        if len(self._store) > 0 and not self.no_warning:
            warnings.warn(f'Leaking {self._store.keys()} in the image feature store')
