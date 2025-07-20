import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, bias=True, initial_range=0.02):
        super().__init__()
        self.initial_range = initial_range
        self.bias = bias

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        if self.bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users, 1))
            self.item_bias = nn.Parameter(torch.zeros(num_items, 1))
            self.global_bias = nn.Parameter(torch.zeros(1))
        self.head = nn.Linear(emb_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=self.initial_range)
        nn.init.normal_(self.item_emb.weight, std=self.initial_range)

    def forward(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        out = self.head(user_emb * item_emb)
        if self.bias:
            out += self.user_bias[user_idx] + self.item_bias[item_idx] + self.global_bias
        return out.squeeze(-1).clip(-2, 2)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers, dropouts, batch_norm=True, initial_range=0.02):
        super().__init__()
        assert len(layers) == len(dropouts) + 1, "len(layers) must be len(dropouts) + 1"
        self.initial_range = initial_range

        self.user_emb = nn.Embedding(num_users, layers[0] // 2)
        self.item_emb = nn.Embedding(num_items, layers[0] // 2)

        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i+1]))
            if batch_norm:
                mlp.append(nn.BatchNorm1d(layers[i+1]))
            mlp.append(nn.ReLU())
            if dropouts:
                mlp.append(nn.Dropout(dropouts[i]))
            
        self.mlp = nn.Sequential(*mlp)

        self.head = nn.Linear(layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=self.initial_range)
        nn.init.normal_(self.item_emb.weight, std=self.initial_range)

    def forward(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        vector = torch.cat([user_emb, item_emb], dim=1)
        vector = self.mlp(vector)

        out = self.head(vector)
        return out.squeeze(-1).clip(-2, 2)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, layers, dropouts, batch_norm=True, initial_range=0.02):
        super().__init__()
        assert len(layers) == len(dropouts) + 1, "len(layers) must be len(dropouts) + 1"
        self.initial_range = initial_range

        self.user_emb_gmf = nn.Embedding(num_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(num_items, emb_dim)

        self.user_emb_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.item_emb_mlp = nn.Embedding(num_items, layers[0] // 2)

        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i+1]))
            if batch_norm:
                mlp.append(nn.BatchNorm1d(layers[i+1]))
            mlp.append(nn.ReLU())
            if dropouts:
                mlp.append(nn.Dropout(dropouts[i]))
            
        self.mlp = nn.Sequential(*mlp)

        self.head = nn.Linear(layers[-1] + emb_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb_gmf.weight, std=self.initial_range)
        nn.init.normal_(self.item_emb_gmf.weight, std=self.initial_range)

        nn.init.normal_(self.user_emb_mlp.weight, std=self.initial_range)
        nn.init.normal_(self.item_emb_mlp.weight, std=self.initial_range)

    def forward(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        # gmf
        user_emb_gmf = self.user_emb_gmf(user_idx)
        item_emb_gmf = self.item_emb_gmf(item_idx)
        gmf_vector = user_emb_gmf * item_emb_gmf

        # mlp
        user_emb_mlp = self.user_emb_mlp(user_idx)
        item_emb_mlp = self.item_emb_mlp(item_idx)

        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
        mlp_vector = self.mlp(mlp_vector)

        # combine
        vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        out = self.head(vector)
        return out.squeeze(-1).clip(-2, 2)

class FM(nn.Module):
    def __init__(self, total_inputs, emb_dim, initial_range=0.02):
        super().__init__()
        self.initial_range = initial_range
        self.total_inputs = total_inputs

        self.emb = nn.Embedding(1 + total_inputs, emb_dim, padding_idx=total_inputs)
        self.linear_layer = nn.Embedding(1 + total_inputs, 1, padding_idx=total_inputs)
        self.bias = nn.Parameter(data=torch.zeros(1))

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=self.initial_range)
        self.emb.weight.data[self.total_inputs].zero_()

    def forward(self, x):
        emb = self.emb(x)
        pow_of_sum = emb.sum(dim=1, keepdim=True).pow(2).sum(dim=2)
        sum_of_pow = emb.pow(2).sum(dim=1, keepdim=True).sum(dim=2)
        out_inter = 0.5 * (pow_of_sum - sum_of_pow)
        out_lin = self.linear_layer(x).sum(1)
        out = out_inter + out_lin + self.bias

        return torch.clamp(out.squeeze(), min=-2, max=2)

class NeuFM(nn.Module):
    def __init__(self, total_inputs, layers, dropouts, batch_norm, initial_range=0.02):
        super().__init__()
        assert len(layers) == len(dropouts) + 1, "len(layers) must be len(dropouts) + 1"
        self.initial_range = initial_range
        self.total_inputs = total_inputs

        self.emb = nn.Embedding(1 + total_inputs, layers[0], padding_idx=total_inputs)
        self.linear_layer = nn.Embedding(1 + total_inputs, 1, padding_idx=total_inputs)
        self.bias = nn.Parameter(data=torch.zeros(1))

        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i+1]))
            if batch_norm:
                mlp.append(nn.BatchNorm1d(layers[i+1]))
            mlp.append(nn.ReLU())
            if dropouts:
                mlp.append(nn.Dropout(dropouts[i]))
            
        self.mlp = nn.Sequential(*mlp)
        
        self.head = nn.Linear(layers[-1], 1)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.emb.weight, std=self.initial_range)
        self.emb.weight.data[self.total_inputs].zero_()

    def forward(self, x):
        emb = self.emb(x)
        pow_of_sum = emb.sum(1).pow(2)
        sum_of_pow = emb.pow(2).sum(1)
        out_inter = 0.5 * (pow_of_sum - sum_of_pow)
        out_inter = self.mlp(out_inter)
        out_inter = self.head(out_inter)
        out_lin = self.linear_layer(x).sum(1)
        out = out_inter + out_lin + self.bias

        return torch.clamp(out.squeeze(), min=-2, max=2)