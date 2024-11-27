import torch
import torch.nn as nn


class MultimodalGatedFusion(nn.Module):
    """
    Multimodal Gated Fusion module.
    """

    def __init__(self,
                 input_dim,
                 n_modalities,
                 hidden_dim=None,
                 gate_input_type='same',
                 gate_output_type='sigmoid-vector'):
        """
        Default constructor.

        Args:
            input_dim (int): Input dimension.
            n_modalities (int): Number of modalities.
            hidden_dim (int): Hidden dimension.
            gate_input_type (str): Gate input type.
            gate_output_type (str): Gate output type.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.input_dim
        self.n_modalities = n_modalities
        self.gate_input_type = gate_input_type
        self.gate_output_type = gate_output_type

        # GATE
        if self.gate_input_type == 'all':
            gate_input_dim = self.input_dim * self.n_modalities
        else:
            assert self.gate_input_type == 'same'
            gate_input_dim = self.input_dim

        if self.gate_output_type == 'sigmoid-scalar':
            gate_act_fun = nn.Sigmoid()
            gate_hidden_dim = 1
        elif self.gate_output_type == 'softmax-scalar':
            gate_act_fun = nn.Identity()
            gate_hidden_dim = 1
        else:
            assert self.gate_output_type == 'sigmoid-vector'
            gate_act_fun = nn.Sigmoid()
            gate_hidden_dim = self.hidden_dim

        self.gate = nn.ModuleList(
            nn.Sequential(nn.Linear(gate_input_dim, gate_hidden_dim), gate_act_fun) for _ in range(self.n_modalities))

        # TANH
        self.tanh = nn.ModuleList(
            nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.Tanh()) for _ in range(self.n_modalities))

    def forward(self, *args):
        """ Forward pass. """
        x = args
        bs = x[-1].size(0)

        tanhs = [self.tanh[i](x[i]) for i in range(self.n_modalities)]

        if self.gate_input_type == 'all':
            x_concat = torch.cat(x, dim=-1)
            gates = [self.gate[i](x_concat) for i in range(self.n_modalities)]
        else:
            assert self.gate_input_type == 'same'
            gates = [self.gate[i](x[i]) for i in range(self.n_modalities)]

        if self.gate_output_type == 'softmax-scalar':
            out = (torch.cat(tanhs, dim=0) *
                   torch.cat(gates, dim=0).view(bs, self.n_modalities).softmax(dim=-1).view(-1, 1)).view(
                       self.n_modalities, bs, -1)
        else:
            out = (torch.cat(tanhs, dim=0) * torch.cat(gates, dim=0)).view(self.n_modalities, bs, -1)

        out = out.sum(dim=0)

        return out
