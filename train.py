from torch import nn, optim

from model_utils import get_image_datasets, get_dataloaders, init_model, train_model, get_cpu_gpu_mode
from get_argparse_info import get_parser_train_info
from checkpoint_utils import save_checkpoint

import torch


if __name__ == "__main__":
    # Parse the arguments specified by the get_parser_train_info() function
    in_args = get_parser_train_info()

    arch = in_args.arch
    epochs = in_args.epochs
    hidden_units = in_args.hidden_units
    learning_rate = in_args.learning_rate
    save_dir = in_args.save_dir
    use_gpu = in_args.gpu
    dir_path = in_args.dir_path[1:] if in_args.dir_path[0] == '/' else in_args.dir_path

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    device_mode = get_cpu_gpu_mode(device)

    # Initializing the model to be used to train
    model = init_model(arch, device, hidden_units)

    # Initialize criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    image_datasets = get_image_datasets(dir_path)
    dataloaders = get_dataloaders(image_datasets)

    # Overview of the architecture and device [cuda or cpu]
    print(f'Arch: {arch} | Device: {device_mode}')

    train_model(model, dataloaders, optimizer, criterion, device, epochs)

    # Check if the user wants to save their trained model as a checkpoint
    print()
    save_checkpoint_input = input('Do you want to save this checkpoint? [y/n]: ').strip().lower()

    if save_checkpoint_input == 'y':
        save_checkpoint(model, image_datasets, arch, hidden_units, save_dir)
    else:
        print('Did not save checkpoint.')
