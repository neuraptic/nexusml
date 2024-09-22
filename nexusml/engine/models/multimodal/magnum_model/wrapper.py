import torch
import torch.nn as nn

from nexusml.engine.models.multimodal.magnum_model.high_level_module import MultimodalGatedFusion
from nexusml.engine.models.multimodal.magnum_model.mid_level_module import GraphPooling
from nexusml.engine.models.multimodal.magnum_model.mid_level_module import Mix


class BottomLevelModule(nn.Module):
    """
    Bottom level module that takes in the tabular, vision and language data
    and returns the embeddings for each modality.
    """

    def __init__(self,
                 d_model: int,
                 language_model: torch.nn.Module = None,
                 vision_model: torch.nn.Module = None,
                 tabular_model: torch.nn.Module = None,
                 language_mapper: torch.nn.Module = None,
                 vision_mapper: torch.nn.Module = None,
                 tabular_mapper: torch.nn.Module = None):
        super().__init__()
        self.d_model = d_model
        self.language_model = language_model
        self.vision_model = vision_model
        self.tabular_model = tabular_model
        self.language_mapper = language_mapper if language_mapper is not None else nn.Identity()
        self.vision_mapper = vision_mapper if vision_mapper is not None else nn.Identity()
        self.tabular_mapper = tabular_mapper if tabular_mapper is not None else nn.Identity()

    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        """ Forward pass of the bottom level module."""
        tab = self.tabular_mapper(self.tabular_model(**tab_data)) if tab_data is not None else None
        vis = self.vision_mapper(self.vision_model(vis_data,
                                                   add_cls_token_output=True)) if vis_data is not None else None
        lan = self.language_mapper(self.language_model(**lan_data,
                                                       add_cls_token_output=True)) if lan_data is not None else None
        return tab, vis, lan


class TopLevelModule(nn.Module):
    """
    Top level module that takes in the embeddings of the tabular, vision and language data
    """

    def __init__(self,
                 d_model,
                 hidden_size,
                 gate_input_type,
                 gate_output_type,
                 k,
                 output_layers: nn.ModuleDict,
                 output_naming_map: dict,
                 modalities=["tabular", "vision", "language"]):
        """
        Constructor

        Args:
            d_model (int): The dimension of the embeddings
            hidden_size (int): The hidden size of the gate
            gate_input_type (str): The type of input to the gate
            gate_output_type (str): The type of output from the gate
            k (int): The number of nearest neighbors to consider
            output_layers (nn.ModuleDict): The output layers
            output_naming_map (dict): The naming map for the output layers
            modalities (list): The list of modalities
        """
        super().__init__()
        if "tabular" in modalities:
            self.tab_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.tab_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)
        if "vision" in modalities:
            self.vis_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.vis_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)
        if "language" in modalities:
            self.lan_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.lan_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)

        self.gate = MultimodalGatedFusion(d_model, len(modalities), hidden_size, gate_input_type, gate_output_type)

        self.classification_heads = output_layers
        self.output_naming_map = output_naming_map

    def forward(self, tab_nodes=None, vis_nodes=None, lan_nodes=None):
        """
        Forward pass of the top level module

        Args:
            tab_nodes (torch.Tensor): The tabular embeddings
            vis_nodes (torch.Tensor): The vision embeddings
            lan_nodes (torch.Tensor): The language embeddings

        Returns:
            dict: The output of the classification heads
        """

        if tab_nodes is not None:
            tab_pool_out = self.tab_graph_pooling(tab_nodes)
            tab_out = self.tab_mix(*tab_pool_out)
        else:
            tab_out = None

        if vis_nodes is not None:
            vis_pool_out = self.vis_graph_pooling(vis_nodes)
            vis_out = self.vis_mix(*vis_pool_out)
        else:
            vis_out = None

        if lan_nodes is not None:
            lan_pool_out = self.lan_graph_pooling(lan_nodes)
            lan_out = self.lan_mix(*lan_pool_out)
        else:
            lan_out = None

        if tab_out is None:
            vis = torch.cat([v.mean(dim=0)[None, :] for v in vis_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None, :] for l in lan_out], dim=0)
            x = (vis, lan)
        elif vis_out is None:
            tab = torch.cat([t.mean(dim=0)[None, :] for t in tab_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None, :] for l in lan_out], dim=0)
            x = (tab, lan)
        elif lan_out is None:
            tab = torch.cat([t.mean(dim=0)[None, :] for t in tab_out], dim=0)
            vis = torch.cat([v.mean(dim=0)[None, :] for v in vis_out], dim=0)
            x = (tab, vis)
        else:
            tab = torch.cat([t.mean(dim=0)[None, :] for t in tab_out], dim=0)
            vis = torch.cat([v.mean(dim=0)[None, :] for v in vis_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None, :] for l in lan_out], dim=0)
            x = (tab, vis, lan)

        x = self.gate(*x)
        x = {self.output_naming_map[k]: v(x) for k, v in self.classification_heads.items()}

        return x


class Magnum(nn.Module):
    """
    MAGNUM Model
    """

    def __init__(
        self,
        bottom_level_module: torch.nn.Module,
        high_level_module: torch.nn.Module,
    ):
        """
        Constructor

        Args:
            bottom_level_module (torch.nn.Module): The bottom level module
            high_level_module (torch.nn.Module): The high level module
        """
        super().__init__()
        self.bottom_level_module = bottom_level_module
        self.high_level_module = high_level_module

    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        """
        Forward pass of the model

        Args:
            tab_data (dict): The tabular data
            vis_data (dict): The vision data
            lan_data (dict): The language data

        Returns:
            dict: The output of the model
        """
        tab, vis, lan = self.bottom_level_module(tab_data, vis_data, lan_data)
        out = self.high_level_module(tab, vis, lan)
        return out
