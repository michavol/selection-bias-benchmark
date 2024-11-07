import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(self, input_dim =180, hidden_dim=100, output_dim=180):
        super(MLP, self).__init__()
        # Define the MLP layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_dim)


    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # Weight initialization
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step] #diffusion_step 143
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    # t_embedding(t). The embedding dimension is 128 in total for every time step t.
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.config = config
        self.channels = config["channels"]
        self.cond_dim = config["cond_dim"] #178
        self.hidden_dim = config["hidden_dim"] #128

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.token_emb_dim = config["token_emb_dim"] if config["mixed"] else 1
        inputdim = 2 * self.token_emb_dim

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        # self.output_projection2 = Conv1d_with_init(self.channels, self.token_emb_dim, 1)

        self.output_projection1 = nn.Linear(self.cond_dim, self.hidden_dim)
        self.output_projection2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.y0_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.y1_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.output_y0 = nn.Linear(self.hidden_dim, 1)
        self.output_y1 = nn.Linear(self.hidden_dim, 1)

        nn.init.zeros_(self.output_projection2.weight)


        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    cond_dim = self.cond_dim,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        # x is the total_input
        # total_input = torch.cat([cond_obs, noisy_target], dim=1) 
        #---------------------------------------------------------------------
        # modify: change cond_info into cond_variable.
        # predicted = self.diffmodel(diff_input, cond_obs, t).to(self.device)


        # total_input torch.Size([8, 1, 178])
        x = x.squeeze(1)
        B, cond_dim = x.shape # (8, 178)
        x = F.relu(x)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            # # # print('layer', layer)
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
        
        ## # print('after layer in residual_layers')

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = F.relu(x)

        # Last layer for y0 and y1 is different.
        y0 = self.y0_layer(x)
        y0 = F.relu(y0)
        y0 = self.output_y0(y0)
        y1 = self.y1_layer(x)
        y1 = F.relu(y1)
        y1 = self.output_y1(y1)

        x = torch.cat((y0,y1), 1)
        # print('forward is finish x.shape', x.shape) 
        # forward is finish x.shape torch.Size([8, 2])
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, cond_dim, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, cond_dim)
        # self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        
        # self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.mid_projection = nn.Linear(cond_dim, cond_dim * 2)

        # self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        # self.cond_projection = nn.Linear(side_dim, 1)
        # self.cond_projection = nn.Linear(33, 1)

        # self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = nn.Linear(cond_dim, cond_dim * 2)

        self.output_projection_for_x = nn.Linear(cond_dim, 2)

        # Temporal Transformer layer
        # self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.time_layer = MLP(input_dim = cond_dim, output_dim=cond_dim)
        # self.time_layer2 = MLP(input_dim = 182, output_dim=182)
        # self.time_layer = MLP(input_dim =512, output_dim=512)
        # Feature Transformer layer
        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = MLP(input_dim = cond_dim, output_dim=cond_dim)
        # self.feature_layer2 = MLP(input_dim = 182, output_dim=182)

    def forward_time(self, y, base_shape):
        # B, channel, K, L = base_shape
        # y = y.permute(0,2,1)
        y = self.time_layer(y)
        # y = y.permute(0,2,1)
        #y = self.time_layer2(y)
        return y 


    def forward_feature(self, y, base_shape):
        # B, channel, K, L = base_shape
        # y = y.permute(0,2,1)
        y = self.feature_layer(y)
        # y = y.permute(0,2,1)
        #y = self.feature_layer2(y)
        return y 
        

    def forward(self, x, cond_info, diffusion_emb):
        # B, channel, K, L = x.shape
        B, cond_dim = x.shape
        base_shape = x.shape
        # x = x.reshape(B, channel, K * L)

        # diffusion_emb is
        # diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
        #     -1
        # )  # (B,channel,1)
        diffusion_emb = self.diffusion_projection(diffusion_emb)
        # # print('diffusion_emb.shape', diffusion_emb.shape) # diffusion_emb.shape torch.Size([8, 180])
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape) 
        y = self.forward_feature(y, base_shape)  
        y = self.mid_projection(y)  

    #%%%%%%%%% not adding cond_info%%%%%%%%%%%%%%%%%%%
        y = y
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        # residual.shape torch.Size([256, 180])
        #skip.shape torch.Size([256, 180])
        return (x + residual) / math.sqrt(2.0), skip
