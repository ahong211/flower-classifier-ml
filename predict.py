import torch
from torch._C import device
from model_utils import predict, get_cpu_gpu_mode

from get_argparse_info import get_parser_test_info
from checkpoint_utils import load_checkpoint
import os

if __name__ == "__main__":
    # Parse the arguments parsed by the get_parser_test_info() function
    in_args = get_parser_test_info()

    image_path = in_args.img_path
    use_gpu = in_args.gpu
    checkpoint_path = in_args.checkpoint
    top_k = in_args.top_k
    category_names = in_args.category_names

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    device_mode = get_cpu_gpu_mode(device)

    # Check if checkpoint path exists and warn user if it is not valid
    if os.path.isfile(checkpoint_path):    
        model = load_checkpoint(checkpoint_path, device)
    else:
        print('Error: The checkpoint path is not valid. Please check your checkpoint path is correct and try again.')
        exit()

    ps, labels = predict(image_path, model, top_k, category_names, device, use_gpu)

    # Print the predicted flower image and probability
    print(f'Inference Mode: {device_mode}')
    print(f'Predicted flower label(s): {labels} | Probability for result(s): {ps}')
