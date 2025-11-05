"""
Loss functions for Pix2Pix training
Including L1 loss and Perceptual loss
"""
import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    Helps preserve high-level features and textures
    """
    def __init__(self, feature_layers=[0, 5, 10, 19, 28], use_input_norm=True):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        # Extract feature layers
        self.feature_extractor = nn.ModuleList()
        for i in range(max(feature_layers) + 1):
            self.feature_extractor.append(vgg[i])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet pretrained models
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """
        Normalize from [-1, 1] to ImageNet stats
        """
        x = (x + 1) / 2  # [-1, 1] to [0, 1]
        x = (x - self.mean) / self.std
        return x
    
    def extract_features(self, x):
        """
        Extract features from multiple layers
        """
        x = self.normalize(x)
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
    
    def forward(self, pred, target):
        """
        Compute perceptual loss
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += nn.functional.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


class GANLoss(nn.Module):
    """
    Standard GAN loss (can be BCE or MSE)
    """
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('GAN mode %s not implemented' % gan_mode)
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss
