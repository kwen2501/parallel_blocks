import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import _mesh_resources


class ColumnWiseLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        tp_mesh=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )

        # Partition information
        # get default device mesh if there's nothing specified
        self._tp_mesh = tp_mesh or _mesh_resources.get_current_mesh()
        self._col_group_size = self._tp_mesh.size()
        self._col_rank = self._tp_mesh.get_local_rank()
        self._col_partition_size = out_features // self._col_group_size

        placement = [Shard(0)]

        # Construct sharded parameters
        # weight only, if bias = False
        # weight and bias, if bias = True
        for name, param in self.named_parameters():
            dtensor = distribute_tensor(param, self._tp_mesh, placement)
            self.register_parameter(
                name,
                Parameter(dtensor),
            )

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, DTensor):
            input = DTensor.from_local(input, self._tp_mesh, [Replicate()])

        return F.linear(input, self.weight, self.bias)


class RowWiseLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        tp_mesh=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )

        # Partition information
        # get default device mesh if there's nothing specified
        self._tp_mesh = tp_mesh or _mesh_resources.get_current_mesh()
        self._row_group_size = self._tp_mesh.size()
        self._row_rank = self._tp_mesh.get_local_rank()
        self._row_partition_size = in_features // self._row_group_size

        placement = [Shard(1)]

        # Construct sharded parameters
        # weight only, if bias = False
        # weight and bias, if bias = True
        for name, param in self.named_parameters():
            dtensor = distribute_tensor(param, self._tp_mesh, placement)
            self.register_parameter(
                name,
                Parameter(dtensor),
            )

    def forward(self, input: Tensor) -> Tensor:
        if not isinstance(input, DTensor):
            input = DTensor.from_local(input, self._tp_mesh, [Shard(-1)])

        return F.linear(input, self.weight, self.bias)
