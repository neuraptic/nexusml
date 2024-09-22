import torch
import torch.nn as nn
import torch_cluster as tc
import torch_geometric

from nexusml.engine.models.multimodal.magnum_model.utils import get_batched_data


class GraphPooling(nn.Module):
    """
    Graph pooling layer
    """

    def __init__(self, d_model, knn_k):
        """
        Constructor

        Args:
            d_model (int): dimension of the input features
            knn_k (int): number of neighbors to consider for pooling
        """
        super().__init__()
        self.d_model = d_model
        self.k = knn_k
        self.edge_pool = torch_geometric.nn.pool.EdgePooling(self.d_model)

    def forward(self, x: torch.Tensor):
        """ Forward pass """
        x_list = []
        edge_index_list = []
        batch_idx_list = []
        for i in range(x.size(0)):
            x_ = x[i]
            edge_index_ = tc.knn_graph(x_, k=self.k, loop=True)
            #edge_index_ = torch.ones(x_.size(0), x_.size(0)).nonzero().T
            x_, edge_index_, batch_idx_, _ = self.edge_pool(x[i],
                                                            edge_index_,
                                                            batch=torch.zeros(x.size(1), device=x_.device).long())
            x_list.append(x_)
            edge_index_list.append(edge_index_)
            batch_idx_list.append(batch_idx_ + i)
        batch_idx = torch.cat(batch_idx_list)

        batch_x, batch_edge_index, _ = get_batched_data(x_list, edge_index_list)

        return batch_x, batch_edge_index, batch_idx


class Mix(nn.Module):
    """
    Mix layer
    """

    def __init__(self, d_model, d_hidden, n_attn_heads):
        """
        Constructor

        Args:
            d_model (int): dimension of the input features
            d_hidden (int): dimension of the hidden layer
            n_attn_heads (int): number of attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_attn_heads = n_attn_heads
        self.layer = torch_geometric.nn.GATv2Conv(self.d_model, self.d_hidden, heads=self.n_attn_heads, concat=False)

    def forward(self, x, edge_index, batch_idx):
        """ Forward pass """
        out = self.layer(x=x, edge_index=edge_index)
        return [out[batch_idx == i] for i in batch_idx.unique()]
