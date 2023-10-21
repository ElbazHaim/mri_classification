import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class PyTorchMLP2(torch.nn.Module):
    def __init__(self, hidden_units, num_features=100, num_classes=2):
        super().__init__()

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(num_features, hidden_unit)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            num_features = hidden_unit

        output_layer = torch.nn.Linear(
            in_features=hidden_units[-1], out_features=num_classes
        )

        all_layers.append(output_layer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.layers(x)
        return logits


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, model=None, hidden_units=None, learning_rate=None):
        super().__init__()

        self.learning_rate = learning_rate
        self.hidden_units = hidden_units

        if model is None:
            self.model = PyTorchMLP2(
                num_features=100, hidden_units=hidden_units, num_classes=2
            )

        else:
            self.model = model

        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
