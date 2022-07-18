from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_scatter import scatter
from torch.autograd import Variable

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from ..inits import glorot, zeros


class GATConvMasked(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.lin_src_sl_1 = Linear(in_channels, 512,
                                   bias=False, weight_initializer='glorot')
        self.lin_src_sl_2 = Linear(512, 256,
                                   bias=False, weight_initializer='glorot')
        self.lin_src_sl_3 = Linear(256, 128,
                                   bias=False, weight_initializer='glorot')

        self.lin_dst_sl_1 = Linear(in_channels, 256,
                                   bias=False, weight_initializer='glorot')
        self.lin_dst_sl_2 = Linear(256, 128,
                                   bias=False, weight_initializer='glorot')

        self.mask_sl_1 = Linear(128, 64, bias=True, weight_initializer='glorot')
        self.mask_sl_2 = Linear(64, 2, bias=True, weight_initializer='glorot')

        self.tau_sl = Parameter(torch.tensor([1.0]), requires_grad=False)

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.num_nodes = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, mask, pre_train, drop,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        # Used for the structure learning
        x_input = x

        self.num_nodes = x.shape[0]

        self.pre_train = pre_train

        self.dropout = drop

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        ####################################################################################
        ####################################################################################

        if not isinstance(mask, Tensor):
            # New structure learning representations
            x_sl = F.elu(self.lin_src_sl_1(x_input))
            x_sl = F.elu(self.lin_src_sl_2(x_sl))
            x_sl = F.elu(self.lin_src_sl_3(x_sl))

            # x_dst_sl = F.elu(self.lin_dst_sl_1(x_input))
            # x_dst_sl = F.elu(self.lin_dst_sl_2(x_dst_sl))

            # Lift and concatenate into their edge positions
            x_src_sl = self.lift(x_sl, edge_index, 0)
            x_dst_sl = self.lift(x_sl, edge_index, 1)
            # edge_reps = torch.cat((x_src_sl, x_dst_sl), dim=1)
            edge_reps = x_src_sl - x_dst_sl  # Subtraction makes the self loops zero, model quickly keeps them all
            # edge_reps = edge_reps / (edge_reps.max() - edge_reps.min())
            # edge_reps = edge_reps - edge_reps.mean()

            # New edge representations
            sl_scores = F.elu(self.mask_sl_1(edge_reps))
            sl_scores = self.mask_sl_2(sl_scores)
            # sl_scores = sl_scores / (sl_scores.max() - sl_scores.min())
            sl_scores = torch.tanh(sl_scores)

            # Generate the new mask
            mask = self.get_mask(sl_scores)
        else:
            sl_scores = None

        # Generate the masked attention coefficients
        alpha = self.sparse_softmax(alpha, edge_index[1], mask)
        # alpha = softmax(alpha, edge_index[1], None, self.num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # alpha = alpha / (torch.sum(mask) / torch.numel(mask))

        ####################################################################################
        ####################################################################################

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, mask, sl_scores

    # Lift the attention coefficients into position
    # This is just broadcasting the node-level attention coefficients into edge-level
    def lift(self, src: torch.Tensor, edge_index: torch.Tensor, dim: int) -> torch.Tensor:

        index = edge_index[dim]
        src = src.index_select(0, index)

        return src

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha = softmax(alpha, index, ptr, size_i)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha  # , mask, scores

    def get_mask(self, scores):

        # probs = F.softmax(scores, dim=1)

        eps = 1e-3
        # probs = probs * (1 - eps)
        # probs = probs + (1 - probs.max()) / 2

        # logits = torch.log(torch.cat((probs, 1 - probs), dim=1))

        # Gumbel noise
        u = torch.rand(scores.shape).float()
        noise = - torch.log(1e-10 - torch.log(u + 1e-10))

        # Add to logits
        logits = scores + ((u - u.mean()) / 10)   # + torch.rand(scores.shape).float() / 10

        # Straight Gumbel way
        soft = F.softmax(logits, dim=1)
        # logits = torch.log(probs)
        # hard = F.gumbel_softmax(logits, tau=self.tau_sl, hard=True)
        # mask = hard[:, 0].unsqueeze(1).expand(hard.shape[0], self.heads)

        # Soft trick way
        _, k = soft.data.max(-1)
        hard = torch.zeros(soft.shape)
        hard = hard.zero_().scatter_(-1, k.view(logits.shape[:-1] + (1,)), 1.0)
        y = hard - soft.data + soft
        mask = y[:, 0].unsqueeze(1).expand(hard.shape[0], self.heads)

        # Just keep all the self loops
        # mask = mask[:-self.num_nodes, :]
        # mask = torch.cat((mask, torch.ones((self.num_nodes, self.heads))))

        if self.pre_train:
            mask = torch.ones(mask.shape)
            # mask[:-self.num_nodes, :] *= 0

        return mask

    # Need to make sure dropout isn't somehow the issue
    def my_dropout(self):
        raise NotImplementedError

    # Masked softmax - returns the softmax over only non-masked edges
    def sparse_softmax(self, alpha: Tensor, index: Tensor, mask: int) -> Tensor:

        # Get each neighborhood minimum
        alpha_min = scatter(alpha, index, 0, dim_size=self.num_nodes, reduce='min')
        alpha_min = alpha_min.index_select(0, index)
        alpha_shifted = alpha + ((1 - mask) * (alpha - (alpha - alpha_min)) - (1 - mask) * 1e-2)

        # Get max
        alpha_max = scatter(alpha_shifted, index, 0, dim_size=self.num_nodes, reduce='max')
        alpha_max = alpha_max.index_select(0, index)

        # Exponentiate
        exp = (alpha - alpha_max).exp()

        # Mask again after exponentiation
        exp = exp * mask

        # Get sum
        alpha_sum = scatter(exp, index, 0, dim_size=self.num_nodes, reduce='sum')
        alpha_sum = alpha_sum.index_select(0, index)

        # Mask one final time after index_select
        alpha_sum = alpha_sum * mask

        # Final output
        alpha = exp / (alpha_sum + (1 - mask))

        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.dropout == 0:
            p = 1 - torch.sum(mask) / torch.numel(mask)
            # alpha = alpha / (1 - p)

        return alpha


    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
