from typing import Union

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data


def get_batched_data(x: Union[list, torch.Tensor],
                     edge_index: Union[list, torch.Tensor],
                     edge_type: Union[list, torch.Tensor] = None):
    """
    Retrieves batched data from input node features, edge indices, and optionally, edge types.

    Args:
        x (Union[list, torch.Tensor]): A list or tensor of node features for each graph in the batch.
        edge_index (Union[list, torch.Tensor]): A list or tensor of edge indices for each graph.
        edge_type (Union[list, torch.Tensor], optional): A list or tensor of edge types for each graph, if applicable.

    Returns:
        torch.Tensor: Batched node features.
        torch.Tensor: Batched edge indices.
        torch.Tensor: Batched edge types (if provided), else None.
    """
    bs = len(x)
    if edge_type is not None:
        batch = Batch.from_data_list(
            [Data(x=x[i], edge_index=edge_index[i], edge_type=edge_type[i]) for i in range(bs)])
        return batch.x, batch.edge_index, batch.edge_type
    else:
        batch = Batch.from_data_list([Data(x=x[i], edge_index=edge_index[i]) for i in range(bs)])
        return batch.x, batch.edge_index, None


def edge_to_adj(edge_index: Union[list, torch.Tensor],
                n_nodes: int,
                edge_type: Union[list, torch.Tensor] = None,
                edge_w: Union[list, torch.Tensor] = None):
    """
    Converts an edge index into an adjacency matrix.

    Args:
        edge_index (Union[list, torch.Tensor]): Edge index, typically of shape (2, num_edges).
        n_nodes (int): The number of nodes in the graph.
        edge_type (Union[list, torch.Tensor], optional): A tensor representing the type of each edge.
        edge_w (Union[list, torch.Tensor], optional): A tensor representing the weight of each edge.

    Returns:
        torch.Tensor: The adjacency matrix of shape (n_nodes, n_nodes).
    """
    assert edge_index.size(0) == 2
    a = torch.zeros(n_nodes, n_nodes, device=edge_index.device)

    if edge_type is not None:
        for t in edge_type.unique():
            idx = edge_index[:, edge_type == t]
            a[idx[0, :], idx[1, :]] = t.item()
        return a

    a[edge_index[0, :], edge_index[1, :]] = 1 if edge_w is None else edge_w
    return a
