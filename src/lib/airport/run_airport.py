from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import cv2
from torchvision import datasets, models, transforms
import time
import os
import copy
from airport.airport_classification import resnet18


class AirportTrainer:
    def __init__(self, config):
        self.config = config

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.config.device)
                    labels = labels.to(self.config.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history


    def run_train_epoch(self):
        # Create training and validation datasets
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.config.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.CenterCrop(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.config.data_dir, x), data_transforms[x]) for x in
                          ['train', 'valid']}
        # Create training and validation dataloaders
        dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.config.batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

        model_ft = resnet18(pretrained=True, num_classes=self.config.num_classes)



        # Send the model to GPU
        model_ft = model_ft.to(self.config.device)

        params_to_update = model_ft.parameters()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        model_ft, hist = self.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=self.config.num_epochs)
        torch.save(model_ft, self.config.model_path)


class AirportClassificationer:
    def __init__(self, config):
        self.config = config
    def test_model(self,  model, dataloaders):

        pass
    def run_test_epoch(self):
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(self.config.input_size),
                transforms.CenterCrop(self.config.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.config.data_dir, x), data_transforms[x]) for x in
                          ['test']}

        model_ft = torch.load(self.config.model_path)
        model_ft.to(self.config.device)
        model_ft.eval()
        for inp, labels in image_datasets.imgs:
            inputs = cv2.imread(inp)
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            print(inp, preds)









