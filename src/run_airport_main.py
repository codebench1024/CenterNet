from __future__ import absolute_import

import _init_paths

from airport.run_airport import AirportTrainer, AirportClassificationer
from airport.train_airport import DefaultConfig

import torch
import os


config = DefaultConfig()

def train():
    trainer = AirportTrainer(config)
    trainer.run_train_epoch()

def test():
    tester = AirportClassificationer(config)
    tester.run_test_epoch()


if __name__ == '__main__':
    train()

