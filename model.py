import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models
from torchvision.models import VGG16_Weights


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Deux convolutions 3x3 + BatchNorm + ReLU
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# UNet Mini

class UNetMini(nn.Module):
    def __init__(self, num_classes=7, base_ch=32):
        super().__init__()
        # Encodeur: 3 niveaux
        self.enc1 = DoubleConv(3, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)

        # Goulot d'étranglement
        self.bottleneck = DoubleConv(base_ch*4, base_ch*8)

        # Décodeur: on remonte et on concatène avec les features de l'encodeur
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch*2, base_ch)

        # Tête: projection vers les classes
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Descente
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Goulot
        b = self.bottleneck(self.pool3(e3))

        # Remontée avec concaténations (skip connections)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Logits (scores non normalisés) [N, num_classes, H, W]
        logits = self.head(d1)
        return logits


# UNet VGG16

class UNetVGG16(nn.Module):
    """UNet avec encodeur VGG16 pré-entraîné"""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # Charger VGG16 pré-entraîné sur ImageNet
        if pretrained:
            vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16 = models.vgg16(weights=None)

        # Extraire les features de VGG16 (sans les couches de classification)
        self.features = vgg16.features

        # Les couches de pooling de VGG16
        self.pool1 = nn.MaxPool2d(2)  # Après conv1_2
        self.pool2 = nn.MaxPool2d(2)  # Après conv2_2
        self.pool3 = nn.MaxPool2d(2)  # Après conv3_3
        self.pool4 = nn.MaxPool2d(2)  # Après conv4_3
        self.pool5 = nn.MaxPool2d(2)  # Après conv5_3

        # Décodeur UNet avec skip connections
        # Up-sampling avec concaténation des features de l'encodeur
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)  # 512 (up) + 512 (skip)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 512, 256)  # 256 (up) + 512 (skip)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 256, 128)  # 128 (up) + 256 (skip)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 128, 64)    # 64 (up) + 128 (skip)

        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(32 + 64, 32)     # 32 (up) + 64 (skip)

        # Tête de classification finale
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encodeur VGG16 avec extraction des features à différents niveaux
        # Conv1 (sans MaxPool de VGG)
        x1 = self.features[0:4](x)    # Après conv1_2: 64 canaux
        x1_pool = self.pool1(x1)

        # Conv2 (ignorer le MaxPool de VGG à l'index 4)
        x2 = self.features[5:9](x1_pool)   # Après conv2_2: 128 canaux
        x2_pool = self.pool2(x2)

        # Conv3 (ignorer le MaxPool de VGG à l'index 9)
        x3 = self.features[10:16](x2_pool) # Après conv3_3: 256 canaux
        x3_pool = self.pool3(x3)

        # Conv4 (ignorer le MaxPool de VGG à l'index 16)
        x4 = self.features[17:23](x3_pool) # Après conv4_3: 512 canaux
        x4_pool = self.pool4(x4)

        # Conv5 (ignorer le MaxPool de VGG à l'index 23)
        x5 = self.features[24:30](x4_pool) # Après conv5_3: 512 canaux
        x5_pool = self.pool5(x5)

        # Décodeur UNet avec skip connections
        # Remontée niveau 4
        up4 = self.upconv4(x5_pool)
        up4 = torch.cat([up4, x5], dim=1)  # Skip connection (512+512=1024)
        up4 = self.dec4(up4)

        # Remontée niveau 3
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, x4], dim=1)  # Skip connection (256+512=768)
        up3 = self.dec3(up3)

        # Remontée niveau 2
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, x3], dim=1)  # Skip connection (128+256=384)
        up2 = self.dec2(up2)

        # Remontée niveau 1
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, x2], dim=1)  # Skip connection (64+128=192)
        up1 = self.dec1(up1)

        # Remontée niveau 0
        up0 = self.upconv0(up1)
        up0 = torch.cat([up0, x1], dim=1)  # Skip connection (32+64=96)
        up0 = self.dec0(up0)

        # Classification finale
        logits = self.final_conv(up0)
        return logits


# DeepLabV3Plus

def build_deeplabv3plus(num_classes=7, encoder_name='resnet50', weights='imagenet'):
    """Construit un modèle DeepLabV3Plus"""
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=3,
        classes=num_classes,
    )
    return model

# HRNet

def build_hrnet(num_classes=7, encoder_name='timm-hrnet_w18', weights='imagenet'):
    """Construit un modèle FPN avec encodeur HRNet ou ResNeSt comme fallback"""
    try:
        model = smp.FPN(
            encoder_name='timm-hrnet_w18',
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        return model
    except Exception:
        print("hrnet indisponible -> utilisation de timm-resnest14d comme fallback")
        model = smp.FPN(
            encoder_name='timm-resnest14d',
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        return model