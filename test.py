# torchrun --nproc-per-node 4 test.py

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor
from linear import ColumnWiseLinear, RowWiseLinear
from embedding import RowWiseEmbedding


# construct a device mesh with available devices (multi-host or single host)
device_mesh = dist.init_device_mesh("cuda", (2, 2), mesh_dim_names=("cp", "tp"))
tp_mesh = device_mesh["tp"]


def test_col_linear():
    x = torch.randn((8, 8))
    col_lin = ColumnWiseLinear(8, 10, bias=False, tp_mesh=tp_mesh)
    y = col_lin(x)
    print(f"\n{y}")


def test_row_linear():
    x = DTensor.from_local(
        torch.ones((1, 4)) * dist.get_rank(),
        tp_mesh,
        [Shard(1)],
    )
    print(f"\n{x=}")

    row_lin = RowWiseLinear(8, 10, bias=False, tp_mesh=tp_mesh)
    y = row_lin(x)
    print(f"\n{y=}")


def test_row_embedding():
    row_emb= RowWiseEmbedding(100, 3, tp_mesh=tp_mesh)
    x = torch.randint(0, 100, (8,))
    y = row_emb(x)
    print(f"\n{y}")


if __name__ == "__main__":
    test_col_linear()
    test_row_linear()
    test_row_embedding()
    dist.destroy_process_group()
