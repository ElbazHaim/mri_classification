import urllib.request
from zipfile import ZipFile
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


class MRIAltzheimerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        transform=None,
        val_train_ratio: int = 0.3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.val_train_ratio = val_train_ratio
        self.train_dir = self.data_dir + "/train"
        self.test_dir = self.data_dir + "/test"

    def setup(self, stage: str):
        if not Path(self.data_dir).is_dir():
            Path(self.data_dir).mkdir(parents=True)

            kaggle_dataset_url = "https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images/download"
            dataset_zip_path = (
                Path(self.data_dir) / "alzheimers-dataset-4-class-of-images.zip"
            )

            urllib.request.urlretrieve(kaggle_dataset_url, dataset_zip_path)

            with ZipFile(dataset_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            subdirectory = Path(self.data_dir) / "Alzheimer_s Dataset"
            for file_path in subdirectory.iterdir():
                if file_path.is_file():
                    file_path.rename(Path(self.data_dir) / file_path.name)

            subdirectory.rmdir()

        self.valtrain_dataset = ImageFolder(self.train_dir, transform=self.transform)
        self.test_dataset = ImageFolder(self.test_dir, transform=self.transform)

        val_size = int(len(self.valtrain_dataset) * self.val_train_ratio)
        train_size = len(self.valtrain_dataset) - val_size

        self.training_dataset, self.validation_dataset = random_split(
            self.valtrain_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
