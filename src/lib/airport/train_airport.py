from __future__ import absolute_import
from airport.run_airport import AirportTrainer, AirportClassificationer

import torch
import os


class DefaultConfig:
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./datadir"
    model_path = os.path.join(data_dir, "resnet18_pretrain_airport.pth")

    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 100

    input_size = 224

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = DefaultConfig()

def train():
    trainer = AirportTrainer(config)
    trainer.run_train_epoch()

def test():
    tester = AirportClassificationer(config)
    tester.run_test_epoch()

if __name__ == '__main__':
    pass
