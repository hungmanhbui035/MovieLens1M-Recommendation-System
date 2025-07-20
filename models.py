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

    def forward(self, user_idx, item_idx):
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        out = self.head(user_emb * item_emb)
        if self.bias:
            out += self.user_bias[user_idx] + self.item_bias[item_idx] + self.global_bias
        return out.squeeze(-1)

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

    def forward(self, user_idx, item_idx):
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)

        vector = torch.cat([user_emb, item_emb], dim=1)
        vector = self.mlp(vector)

        out = self.head(vector)
        return out.squeeze(-1)

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

    def forward(self, user_idx, item_idx):
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
        return out.squeeze(-1)