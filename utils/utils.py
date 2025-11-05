"""
Utility functions for training and inference
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_file, model, optimizer=None, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {checkpoint_file}, epoch: {epoch}")
    return epoch


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image
    Tensor should be in range [-1, 1]
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:  # Batch
        tensor = tensor[0]  # Take first image
    
    numpy_image = tensor.cpu().detach().numpy()
    numpy_image = numpy_image.transpose(1, 2, 0)
    
    # Convert to PIL Image
    numpy_image = (numpy_image * 255).astype(np.uint8)
    return Image.fromarray(numpy_image)


def save_image(tensor, path):
    """Save tensor as image"""
    image = tensor_to_image(tensor)
    image.save(path)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
