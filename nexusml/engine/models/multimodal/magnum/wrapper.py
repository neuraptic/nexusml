import torch
import torch.nn as nn

from nexusml.engine.models.multimodal.magnum.high_level_module import MultimodalGatedFusion
from nexusml.engine.models.multimodal.magnum.mid_level_module import GraphPooling
from nexusml.engine.models.multimodal.magnum.mid_level_module import Mix


class BottomLevelModule(nn.Module):
    """
    Bottom-level module that processes tabular, vision, and language data and returns their respective embeddings.
    """

    def __init__(self,
                 d_model: int,
                 language_model: torch.nn.Module = None,
                 vision_model: torch.nn.Module = None,
                 tabular_model: torch.nn.Module = None,
                 language_mapper: torch.nn.Module = None,
                 vision_mapper: torch.nn.Module = None,
                 tabular_mapper: torch.nn.Module = None):
        """
        Initializes the BottomLevelModule with models and mappers for each modality.

        Args:
            d_model (int): The hidden dimension of the embeddings.
            language_model (torch.nn.Module): The model used for processing language data.
            vision_model (torch.nn.Module): The model used for processing vision data.
            tabular_model (torch.nn.Module): The model used for processing tabular data.
            language_mapper (torch.nn.Module, optional): A mapper for language data embeddings.
            vision_mapper (torch.nn.Module, optional): A mapper for vision data embeddings.
            tabular_mapper (torch.nn.Module, optional): A mapper for tabular data embeddings.
        """
        super().__init__()
        self.d_model = d_model
        self.language_model = language_model
        self.vision_model = vision_model
        self.tabular_model = tabular_model
        self.language_mapper = language_mapper if language_mapper is not None else nn.Identity()
        self.vision_mapper = vision_mapper if vision_mapper is not None else nn.Identity()
        self.tabular_mapper = tabular_mapper if tabular_mapper is not None else nn.Identity()

    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        """
        Forward pass of the bottom-level module, which processes each modality independently and returns embeddings.

        Args:
            tab_data (dict, optional): Input data for tabular modality.
            vis_data (dict, optional): Input data for vision modality.
            lan_data (dict, optional): Input data for language modality.

        Returns:
            tuple: Embeddings for tabular, vision, and language data.
        """
        tab = self.tabular_mapper(self.tabular_model(**tab_data)) if tab_data is not None else None
        vis = self.vision_mapper(self.vision_model(vis_data,
                                                   add_cls_token_output=True)) if vis_data is not None else None
        lan = self.language_mapper(self.language_model(**lan_data,
                                                       add_cls_token_output=True)) if lan_data is not None else None
        return tab, vis, lan


class TopLevelModule(nn.Module):
    """
    Top-level module that processes embeddings from tabular, vision, and language data and applies
    further graph pooling and mixing before passing them to the classification heads.
    """

    def __init__(self,
                 d_model: int,
                 hidden_size: int,
                 gate_input_type: str,
                 gate_output_type: str,
                 k: int,
                 output_layers: nn.ModuleDict,
                 output_naming_map: dict,
                 modalities: list = ["tabular", "vision", "language"]):
        """
        Initializes the TopLevelModule with graph pooling, mixing layers, and a multimodal gated fusion mechanism.

        Args:
            d_model (int): The dimension of the embeddings.
            hidden_size (int): The hidden size of the gated fusion layer.
            gate_input_type (str): The input type for the gated fusion layer.
            gate_output_type (str): The output type for the gated fusion layer.
            k (int): The number of nearest neighbors for graph pooling.
            output_layers (nn.ModuleDict): Classification heads for each output.
            output_naming_map (dict): A mapping from output layer names to their respective layers.
            modalities (list): A list of modalities, defaulting to ["tabular", "vision", "language"].
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
        Forward pass of the top-level module, processing the modality-specific embeddings and
        fusing them through a gated fusion mechanism.

        Args:
            tab_nodes (torch.Tensor, optional): The embeddings from tabular data.
            vis_nodes (torch.Tensor, optional): The embeddings from vision data.
            lan_nodes (torch.Tensor, optional): The embeddings from language data.

        Returns:
            dict: The outputs from the classification heads, mapped by output names.
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
    MAGNUM Model, a multi-modal architecture combining tabular, vision, and language modalities
    through a hierarchical fusion mechanism involving bottom-level and high-level modules.
    """

    def __init__(
        self,
        bottom_level_module: torch.nn.Module,
        high_level_module: torch.nn.Module,
    ):
        """
        Initializes the MAGNUM model with bottom-level and high-level modules.

        Args:
            bottom_level_module (torch.nn.Module): The module responsible for processing raw modality data.
            high_level_module (torch.nn.Module): The module responsible for fusing modality embeddings.
        """
        super().__init__()
        self.bottom_level_module = bottom_level_module
        self.high_level_module = high_level_module

    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        """
        Forward pass of the MAGNUM model, processing tabular, vision, and language data through the
        bottom-level and high-level modules to generate predictions.

        Args:
            tab_data (dict, optional): The input tabular data.
            vis_data (dict, optional): The input vision data.
            lan_data (dict, optional): The input language data.

        Returns:
            dict: The output from the model.
        """
        tab, vis, lan = self.bottom_level_module(tab_data, vis_data, lan_data)
        out = self.high_level_module(tab, vis, lan)
        return out
