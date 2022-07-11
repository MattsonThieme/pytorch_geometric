from typing import Optional

from torch import Tensor
from torch_scatter import gather_csr, scatter, segment_csr

from .num_nodes import maybe_num_nodes


def softmax_sign(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_mean = scatter(src, index, dim, dim_size=N, reduce='mean')
        src_mean = src_mean.index_select(dim, index)

        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_min = scatter(src, index, dim, dim_size=N, reduce='min')
        src_span = src_max - src_min
        src_span = src_span.index_select(dim, index)

        out = (src - src_mean) / src_span

        out_sum = scatter(out.abs(), index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    # signs = src / src.abs()
    # out = out * signs

    return out  # / (out_sum + 1e-16)
