import torch
import torch.nn as nn
import torch_cluster as tc
import torch_geometric

from nexusml.engine.models.multimodal.magnum.utils import get_batched_data


class GraphPooling(nn.Module):
    """
    Graph pooling layer
    """

    def __init__(self, d_model, knn_k):
        """
        Initializes the GraphPooling layer.

        Args:
            d_model (int): The dimensionality of the input features for each node.
            knn_k (int): The number of nearest neighbors to consider for pooling.
        """
        super().__init__()
        self.d_model = d_model
        self.k = knn_k
        self.edge_pool = torch_geometric.nn.pool.EdgePooling(self.d_model)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): A batch of input node features of shape (batch_size, num_nodes, d_model).

        Returns:
            torch.Tensor: Pooled node features.
            torch.Tensor: Edge indices after pooling.
            torch.Tensor: Batch indices for the pooled nodes.
        """
        x_list = []
        edge_index_list = []
        batch_idx_list = []
        for i in range(x.size(0)):
            x_ = x[i]
            edge_index_ = tc.knn_graph(x_, k=self.k, loop=True)
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
        Initializes the Mix layer.

        Args:
            d_model (int): The dimensionality of the input node features.
            d_hidden (int): The dimensionality of the hidden layer output.
            n_attn_heads (int): The number of attention heads for the GAT layer.
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_attn_heads = n_attn_heads
        self.layer = torch_geometric.nn.GATv2Conv(self.d_model, self.d_hidden, heads=self.n_attn_heads, concat=False)

    def forward(self, x, edge_index, batch_idx):
        """
        Forward pass.

        Args:
            x (torch.Tensor): The input node features of shape (num_nodes, d_model).
            edge_index (torch.Tensor): The edge indices connecting the nodes.
            batch_idx (torch.Tensor): The batch indices indicating which nodes belong to which graph in the batch.

        Returns:
            List[torch.Tensor]: A list of node feature outputs for each graph in the batch.
        """
        out = self.layer(x=x, edge_index=edge_index)
        return [out[batch_idx == i] for i in batch_idx.unique()]
