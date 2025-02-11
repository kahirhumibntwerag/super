import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.transforms import Normalize

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model
        self.vgg = vgg19(pretrained=True).features
        self.vgg = self.vgg.to('cuda' if torch.cuda.is_available() else 'cpu')
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Define layers to use for feature extraction
        self.layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '17': 'relu3_3',
            '26': 'relu4_3'
        }

        # Normalization for VGG19
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_features(self, image):
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

    def forward(self, target_image, output_image):
        # Preprocess the images
        #target_image = target_image/target_image.max()
        target_image = target_image.repeat(1,3,1,1)

        #output_image = output_image/output_image.max()
        output_image = output_image.repeat(1,3,1,1)

        target_image = self.normalize(target_image)
        output_image = self.normalize(output_image)

        # Extract features
        target_features = self.get_features(target_image)
        output_features = self.get_features(output_image)

        # Calculate Perceptual Loss
        loss = 0.0
        for layer in self.layers.values():
            loss += torch.nn.functional.l1_loss(target_features[layer], output_features[layer])

        return loss
    



class CombinedLoss(nn.Module):
    def __init__(self, discriminator, alpha=0.001, beta=1.0, gamma=0.01):
        super().__init__()
        self.mse_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCELoss()
        self.perceptual_loss = PerceptualLoss()
        self.discriminator = discriminator
        self.alpha = alpha  # Weight for the adversarial loss
        self.beta = beta    # Weight for the perceptual loss
        self.gamma = gamma
    def forward(self, high_res_fake, high_res_real):
        # MSE Loss
        mse_loss_value = self.mse_loss(high_res_fake, high_res_real)

        # Perceptual Loss
        perceptual_loss_value = self.perceptual_loss(high_res_fake, high_res_real)

        # Adversarial Loss
        predictions_fake = self.discriminator(high_res_fake)
        labels_real = torch.ones_like(predictions_fake, device=predictions_fake.device)
        adversarial_loss_value = self.adversarial_loss(predictions_fake, labels_real)

        # Combined Loss
        combined_loss_value = self.alpha * adversarial_loss_value + self.beta * perceptual_loss_value + self.gamma*mse_loss_value
        return combined_loss_value, mse_loss_value, adversarial_loss_value, perceptual_loss_value
