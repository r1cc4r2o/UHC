import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LinearWeightBlock(nn.Module):
    def __init__(self, dim_embedding = 256, n_head = 2) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding

        self.linear = nn.Linear(dim_embedding, dim_embedding//16)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_embedding//16, nhead=n_head), num_layers=2
        )

        self.bilinear = nn.Bilinear(dim_embedding//16, dim_embedding//16, dim_embedding//16)

        self.dropout = nn.Dropout(0.1)

        self.gelu = nn.GELU()

        self.layer_norm = nn.LayerNorm(dim_embedding//16)

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        x_0 = self.transformer_encoder(x)
        x = self.bilinear(x_0 + self.dropout(x_0), x)
        x = self.gelu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, dim_embedding = 256, n_head = 2, n_class = 15) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding
        self.n_class = n_class
        self.number_of_amino_acids = 21

        self.lernable_emb = nn.Embedding(self.number_of_amino_acids, dim_embedding)

        self.linear_weight_block = LinearWeightBlock(dim_embedding, n_head)

        self.flatten = nn.Flatten(start_dim=1, end_dim=- 1)

        # 1966 is the lenght of the sequence
        self.linear_transformation = nn.Linear((dim_embedding//16) * 1966 , dim_embedding//32)

        self.layer_norm = nn.LayerNorm(dim_embedding//32)

        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding//32, dim_embedding//16),
            nn.GELU(),
            nn.LayerNorm(dim_embedding//16),
            nn.Linear(dim_embedding//16, n_class)
        )

    def forward(self, x):
        x = self.lernable_emb(x.type(torch.long))
        x = self.linear_weight_block(x)
        x = self.flatten(x)
        x = self.linear_transformation(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, dim_embedding = 256, n_head = 2, n_class = 15) -> None:
        super().__init__()

        self.model = Classifier(dim_embedding, n_head, n_class)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_class)
        self.f1 = torchmetrics.F1Score(task="multiclass",num_classes=n_class)
        self.precision = torchmetrics.Precision(task="multiclass",num_classes=n_class)
        self.recall = torchmetrics.Recall(task="multiclass",num_classes=n_class)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # y = y.long()

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y = torch.argmax(y, dim = -1)

        self.log("train_loss", loss)
        # self.log("train_acc", torch.sum(torch.argmax(y_hat, dim=-1) == torch.argmax(y, dim=-1))/x.shape[0])
        # self.log("train_acc", self.accuracy(torch.argmax(y_hat, dim=-1),torch.argmax(y, dim=-1)))
        self.log("train_acc", self.accuracy(y_hat, y))
        self.log("train_f1", self.f1(y_hat, y))
        self.log("train_precision", self.precision(y_hat, y))
        self.log("train_recall", self.recall(y_hat, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # y = y.long()

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y = torch.argmax(y, dim = -1)

        self.log("val_loss", loss)
        # self.log("train_acc", torch.sum(torch.argmax(y_hat, dim=-1) == torch.argmax(y, dim=-1))/x.shape[0])
        # self.log("train_acc", self.accuracy(torch.argmax(y_hat, dim=-1),torch.argmax(y, dim=-1)))
        self.log("val_acc", self.accuracy(y_hat, y))
        self.log("val_f1", self.f1(y_hat, y))
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

