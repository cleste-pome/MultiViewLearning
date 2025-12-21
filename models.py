import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.dims = dims
        models = []
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i != len(self.dims) - 2:
                models.append(nn.ReLU())
            else:
                models.append(nn.Dropout(p=0.5))
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.dims = dims
        models = []
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i == len(self.dims) - 2:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Sigmoid())
            else:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim),
            nn.LeakyReLU(),
            nn.Linear(self.feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


def discriminator_loss(real_out, fake_out, lambda_dis=1):
    real_loss = nn.BCEWithLogitsLoss()(real_out, torch.ones_like(real_out))
    fake_loss = nn.BCEWithLogitsLoss()(fake_out, torch.zeros_like(fake_out))
    return lambda_dis * (real_loss + fake_loss)


class MvAEModel(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims, num_classes):
        super().__init__()
        self.input_dims = input_dims
        self.view_num = view_num
        self.out_dims = out_dims
        self.h_dims = h_dims
        self.num_classes = num_classes
        self.discriminators = nn.ModuleList()

        for v in range(view_num):
            self.discriminators.append(Discriminator(out_dims))

        h_dims_reverse = list(reversed(h_dims))
        self.encoders_specific = nn.ModuleList()
        self.decoders_specific = nn.ModuleList()

        for v in range(self.view_num):
            self.encoders_specific.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            self.decoders_specific.append(Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]))

        d_sum = sum(input_dims)
        self.encoder_share = Encoder([d_sum] + h_dims + [out_dims])

        # 添加分类头
        self.classifier2 = nn.Linear(out_dims, num_classes)
        # TODO 添加分类头
        hidden_dim = out_dims * (view_num + 1)
        mid = min(256, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid, num_classes),
        )

        # TODO 残差分类头
        # self.classifier_in = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, mid),
        #     nn.GELU(),
        # )
        # # 残差块：增强表达但不容易训练崩/过拟合
        # self.classifier_block = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(mid, mid),
        #     nn.GELU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(mid, mid),
        # )
        # self.classifier_out = nn.Sequential(
        #     nn.LayerNorm(mid),
        #     nn.Linear(mid, num_classes),
        # )

        # LayerNorm
        self.block_norm = nn.LayerNorm(out_dims)

        # Gate control
        self.gate = nn.Sequential(
            nn.Linear(out_dims, out_dims // 2 if out_dims >= 2 else 1),
            nn.GELU(),
            nn.Linear(out_dims // 2 if out_dims >= 2 else 1, 1)
        )

    def discriminators_loss(self, hidden_specific, i, LAMB_DIS=1):
        discriminate_loss = 0.
        for j in range(self.view_num):
            if j != i:
                real_out = self.discriminators[i](hidden_specific[i])
                fake_out = self.discriminators[i](hidden_specific[j])
                discriminate_loss += discriminator_loss(real_out, fake_out, LAMB_DIS)
        return discriminate_loss

    def forward(self, x_list):
        x_total = torch.cat(x_list, dim=-1)
        hidden_share = self.encoder_share(x_total)
        recs = []
        hidden_specific = []

        for v in range(self.view_num):
            x = x_list[v]
            hidden_specific_v = self.encoders_specific[v](x)
            hidden_specific.append(hidden_specific_v)
            hidden_v = torch.cat((hidden_share, hidden_specific_v), dim=-1)
            rec = self.decoders_specific[v](hidden_v)
            recs.append(rec)

        # hidden_list = [hidden_share] + hidden_specific
        hidden_list = [self.block_norm(hidden_share)] + [self.block_norm(h) for h in hidden_specific] # TODO LayerNorm
        # g = [self.gate(h) for h in hidden_list]  # TODO Gate control
        # alpha = torch.softmax(torch.cat(g, dim=1), dim=1)
        # hidden_list = [h * alpha[:, i:i + 1] for i, h in enumerate(hidden_list)]
        hidden = torch.cat(hidden_list, dim=-1)

        # TODO 分类输出
        # class_output = self.classifier2(hidden_share)
        class_output = self.classifier(hidden)
        # TODO 残差分类头
        # h = self.classifier_in(hidden)
        # h = h + self.classifier_block(h)
        # class_output = self.classifier_out(h)

        return hidden_share, hidden_specific, hidden, recs, class_output
