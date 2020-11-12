from torchvision import datasets, transforms, models
from torch import nn

from collections import OrderedDict
from PIL import Image

import numpy as np
import torch
import time
import json
import os


def init_model(arch, device, hidden_units):
    """Initialize the model to be created based on the architecture given"""

    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 256)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout()),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    model.to(device)

    return model


def get_image_datasets(data_dir='flowers'):
    """Returns the image datasets as a dictionary for the training and validation phases"""

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    return image_datasets


def get_dataloaders(image_datasets):
    """Returns the dataloaders as a dictionary for the training and validation phases"""

    batch_size = 64

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
    }

    return dataloaders


def train_model(model, dataloaders, optimizer, criterion, device, num_epochs=5):
    """Training logic for the model based on the training and validation phases"""

    start = time.time()
    print()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} / {num_epochs}')

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            accuracy = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logps = model(inputs)
                    loss = criterion(logps, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            phase_loss = running_loss/len(dataloaders[phase])
            phase_accuracy = accuracy/len(dataloaders[phase])

            print(f'{phase} Loss: {phase_loss:.4f} Accuracy: {phase_accuracy:.4f}')

        print()

    total_time = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))


def get_cpu_gpu_mode(device):
    """Returns a string to determine the device type"""
    return 'GPU' if device == torch.device('cuda') else 'CPU'


def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model. Returns the updated image"""

    img = Image.open(image_path)

    size = 256, 256
    img.thumbnail(size)
    width, height = img.size

    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    img = img.crop((left, top, right, bottom))

    img = np.array(img) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = img.transpose((2, 0, 1))
    return img


def predict(image_path, model, top_k, category_names, device, use_gpu):
    """Predicts the flower and probability based on the image given and the loaded model"""

    # Check if the category name argparse input is valid
    if os.path.isfile(category_names):
        with open(category_names, 'r') as f:
            category_json = json.load(f)
    else:
        print("The category name file does not exist. Defaulting to 'cat_to_name.json'")

        with open('cat_to_name.json', 'r') as f:
            category_json = json.load(f)

    new_image = process_image(image_path)

    # Converts the tensor type based off the use_gpu flag
    img_tensor = torch.from_numpy(new_image).type(torch.cuda.FloatTensor if use_gpu else torch.FloatTensor)
    img_input = img_tensor.unsqueeze(0)

    # We will use either 'cuda' or 'cpu' for the model based off the device
    model.to(device)
    ps = torch.exp(model(img_input))

    flower_ps, labels = ps.topk(top_k)
    flower_ps = flower_ps.tolist()[0]
    labels = labels.tolist()[0]

    # Inverting the model.class_to_idx values and keys
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    # Output the flower labels based off the category_json variable
    flower_labels = []
    for label in labels:
        flower_labels.append(category_json[idx_to_class[label]])

    return flower_ps, flower_labels
