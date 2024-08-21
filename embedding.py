from typing import Optional

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import _mesh_resources


class RowWiseEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
        tp_mesh=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            device,
            dtype,
        )

        # Partition information
        # get default device mesh if there's nothing specified
        self._tp_mesh = tp_mesh or _mesh_resources.get_current_mesh()
        self._row_group_size = self._tp_mesh.size()

        # rowwise shard embedding.weight is Shard(0)
        for name, param in self.named_parameters():
            dist_param = Parameter(distribute_tensor(param, tp_mesh, [Shard(0)]))
            self.register_parameter(name, dist_param)

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, DTensor):
            input = DTensor.from_local(input, self._tp_mesh, [Replicate()])
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)
