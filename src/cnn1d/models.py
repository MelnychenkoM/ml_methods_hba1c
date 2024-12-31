import os
import numpy as np
import torch
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as L

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.cli import LightningCLI
from dataset import SpectraDataset, SpectraDataModule

from datetime import datetime

class InceptionModule(nn.Module):
    pass


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=360)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(in_features=360, out_features=180)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(in_features=180, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CNN1DModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DModel, self).__init__()
        self.input_size = input_size

        self.conv1d1 = nn.Conv1d(in_channels=1, 
                                 out_channels=64, 
                                 kernel_size=5, 
                                 stride=1, 
                                 padding=5)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1d2 = nn.Conv1d(in_channels=64, 
                                 out_channels=64, 
                                 kernel_size=5, 
                                 stride=1, 
                                 padding=5)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        flatten_size = self._calculate_flatten_size()
        self.fc = FullyConnectedLayer(flatten_size)

    def _calculate_flatten_size(self):
        dummy_input = torch.zeros(1, 1, self.input_size)
        with torch.no_grad():
            dummy_output = self.max_pool1(F.relu(self.batch_norm1(self.conv1d1(dummy_input))))
            dummy_output = self.max_pool2(F.relu(self.batch_norm2(self.conv1d2(dummy_output))))
        return dummy_output.numel()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.batch_norm1(self.conv1d1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.batch_norm2(self.conv1d2(x)))
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Lightning1DCNNModel(L.LightningModule):
    def __init__(self, input_size, learning_rate=1e-3):
        super(Lightning1DCNNModel, self).__init__()
        self.model = CNN1DModel(input_size)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        self.r2_metric = torchmetrics.R2Score()
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.pearson_corr = torchmetrics.PearsonCorrCoef()
        self.mae_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        r2 = self.r2_metric(y_pred, y)
        rmse = self.rmse_metric(y_pred, y)
        r = self.pearson_corr(y_pred, y)
        mae = self.mae_metric(y_pred, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_r2', r2, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('val_pearson_r', r, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y)

        r2 = self.r2_metric(y_pred, y)
        rmse = self.rmse_metric(y_pred, y)
        r = self.pearson_corr(y_pred, y)
        mae = self.mae_metric(y_pred, y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_r2', r2, on_epoch=True)
        self.log('test_rmse', rmse, on_epoch=True)
        self.log('test_pearson_r', r, on_epoch=True)
        self.log('test_mae', mae, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    

def generate_run_name(lr, batch_size, timestamp=True):
    name = f"lr_{lr}_bs_{batch_size}"
    if timestamp:
        name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return name
    

def main():
    torch.cuda.empty_cache()
    torch.manual_seed(32)

    input_size = 3319
    learning_rate = 1e-4
    batch_size = 32
    max_epochs = 300

    data_module = SpectraDataModule(data_folder='../../data/train_test_cnn/', batch_size=batch_size)

    model = Lightning1DCNNModel(input_size=input_size, learning_rate=learning_rate)

    run_name = generate_run_name(learning_rate, batch_size)
    mlf_logger = MLFlowLogger(experiment_name="1D_CNN_Regression", 
                              save_dir="./mlruns",
                              run_name=run_name
                              )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print("[1DCNN] Training complete. Checkpoints and logs saved.")

if __name__ == "__main__":
    main()
