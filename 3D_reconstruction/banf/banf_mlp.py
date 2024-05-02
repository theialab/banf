import torch
from torch import nn
import numpy as np

class Geo_MLP(nn.Module):
    def __init__(self, field_object, init_factor=1):
        super().__init__()
        dims = [field_object.config.hidden_dim for _ in range(field_object.config.num_layers)]
        in_dim = 3 + field_object.position_encoding.get_out_dim() + field_object.encoding1.n_output_dims
        dims = [in_dim] + dims + [1+field_object.config.geo_feat_dim]
        field_object.num_layers = len(dims)

        field_object.skip_in = [4]
        self.num_layers = field_object.num_layers
        self.softplus = nn.Softplus(beta=100)
        self.skip_in = field_object.skip_in

        for l in range(0, field_object.num_layers - 1):
            if l + 1 in field_object.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if field_object.config.geometric_init:
                if l == field_object.num_layers - 2:
                    if not field_object.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) * init_factor / np.sqrt(dims[l]), std=1e-10)
                        torch.nn.init.constant_(lin.bias, -field_object.config.bias * init_factor)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) * init_factor / np.sqrt(dims[l]), std=1e-10)
                        torch.nn.init.constant_(lin.bias, field_object.config.bias * init_factor)
                    
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if field_object.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(l), lin)

    def forward(self, inputs):
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x 

class Color_MLP(nn.Module):
    def __init__(self, field_object, init_factor=1, color_continuous=False):
        super().__init__()
        # view dependent color network
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.rgb_padding = field_object.config.rgb_padding
        self.init_factor = init_factor

        dims = [field_object.config.hidden_dim_color for _ in range(field_object.config.num_layers_color)]

        # point, view_direction, normal, feature, embedding
        point_dim = field_object.position_encoding.get_out_dim() if field_object.config.encoding_type == "grid" else 3
        in_dim = (
            point_dim
            + field_object.direction_encoding.get_out_dim()
            + 3 ### normal
            + field_object.config.geo_feat_dim
        )
        if color_continuous:
            in_dim += field_object.encoding1.n_output_dims
        if field_object.use_average_appearance_embedding:
            in_dim += field_object.embedding_appearance.get_out_dim()

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)
            if l == (self.num_layers_color - 2):
                if init_factor == 1:
                    torch.nn.init.kaiming_uniform_(lin.weight.data)
                elif init_factor == 0:
                    torch.nn.init.normal_(lin.weight, mean=0, std=1e-10)
                    # torch.nn.init.zeros_(lin.weight.data)
                else:
                    assert 0, "Need proper init for this value of init_factor"

            if field_object.config.weight_norm and init_factor == 1:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)
    
    def forward(self, h):
        """compute colors"""
        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        return h