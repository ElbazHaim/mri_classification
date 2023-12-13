import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding=padding
        )
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        return x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_classes: int):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(input_size, output_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(output_size, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


def get_conv_blocks(in_channels: int = 1, num_conv_blocks: int = 1):
    feature_extractor_layers = []
    out_channels = 16

    for _ in range(num_conv_blocks):
        feature_extractor_layers.extend(
            [
                ConvBlock(in_channels, out_channels, 5, padding=1),
            ]
        )
        in_channels = out_channels
        out_channels *= 2

    return nn.Sequential(*feature_extractor_layers)


class PlLeNet(pl.LightningModule):
    def __init__(
        self,
        input_shape: tuple = (1, 188, 156),
        num_classes: int = 4,
        num_conv_blocks: int = 1,
        learning_rate: int = 0.001,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.example_input_array = torch.randn(32, 1, input_shape[-2], input_shape[-1])

        self.feature_extractor = get_conv_blocks(input_shape[0], num_conv_blocks)

        _x = self.feature_extractor(self.example_input_array)
        conv_output_size = _x.view(_x.size(0), -1).shape[-1]
        self.classifier = FullyConnectedBlock(conv_output_size, 150, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)

        acc = torchmetrics.functional.accuracy(
            y_hat, y, task="multiclass", num_classes=4
        )
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        acc = torchmetrics.functional.accuracy(
            y_hat, y, task="multiclass", num_classes=4
        )
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

        acc = torchmetrics.functional.accuracy(
            y_hat, y, task="multiclass", num_classes=4
        )
        self.log("test_acc", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
