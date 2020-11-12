import argparse

parser = argparse.ArgumentParser()


def get_parser_train_info():
    parser.add_argument('dir_path', type=str, default='flowers',
                        help='directory path to be used')
    parser.add_argument('--save_dir', type=str, default='save_checkpoints',
                        help='save path to the directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='architecture to train model')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='floating number used for the learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024,
                        help='hidden units to be trained in our model')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train the model')
    parser.add_argument('--gpu', action='store_true',
                        help='flag to enable model to use gpu')

    return parser.parse_args()


def get_parser_test_info():
    parser.add_argument('img_path', type=str,
                        help='image path for the script to predict')
    parser.add_argument('checkpoint', type=str, default='save_checkpoints/checkpoint.pth',
                        help='specify the checkpoint to load')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k amount of probabilities to predict')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='file to load for the category names')
    parser.add_argument('--gpu', action='store_true',
                        help='flag to enable model to use gpu')

    return parser.parse_args()
