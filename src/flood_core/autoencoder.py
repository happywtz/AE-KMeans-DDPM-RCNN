"""Convolutional autoencoder for paired 48-hour Q/P sequences."""

from typing import Tuple

import torch
from torch import nn


class HistoricalFiveDimensionalAE(nn.Module):
    """Exact architecture of the archived Variable_AutoEncoder."""

    def __init__(self) -> None:
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 10, kernel_size=3, stride=2),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(10, 5, kernel_size=3, stride=1),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(5, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 256, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(32, 2, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.Encoder(values)
        return self.Decoder(latent), latent


Autoencoder = HistoricalFiveDimensionalAE
