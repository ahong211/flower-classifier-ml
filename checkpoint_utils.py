from model_utils import init_model

import torch
import os


def save_checkpoint(model, image_datasets, arch, hidden_units, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    if not os.path.isdir(save_dir):
        print(f'{save_dir} is not a directory. Creating a new directory with name: {save_dir}')
        os.mkdir(save_dir)

    checkpoint_file = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    class_to_idx = checkpoint['class_to_idx']
    model_state_dict = checkpoint['state_dict']

    model = init_model(arch, device, hidden_units)

    model.class_to_idx = class_to_idx
    model.load_state_dict(model_state_dict)

    return model
