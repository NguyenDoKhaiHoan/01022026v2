"""
Evolution Module - Feature Evolution Subnetwork
Combines Transformation Module (encoder) and Recovery Module (decoder)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=[64, 128, 256]):
        super(TransformationModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        features = []
        x1 = self.conv1(x)
        features.append(x1)
        x2 = self.conv2(x1)
        features.append(x2)
        x3 = self.conv3(x2)
        features.append(x3)
        return features, x3


class RecoveryModule(nn.Module):
    def __init__(self, encoded_channels=256, hidden_channels=[128, 64, 32], out_channels=3):
        super(RecoveryModule, self).__init__()
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(encoded_channels, hidden_channels[0], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[0], hidden_channels[1], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[1], hidden_channels[2], 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Conv2d(hidden_channels[2], out_channels, kernel_size=3, padding=1)
        
    def forward(self, encoded):
        reconstructions = []
        x1 = self.deconv1(encoded)
        reconstructions.append(x1)
        x2 = self.deconv2(x1)
        reconstructions.append(x2)
        x3 = self.deconv3(x2)
        reconstructions.append(x3)
        output = self.output_conv(x3)
        return reconstructions, output


class FeatureEvolutionSubnetwork(nn.Module):
    """
    Feature Evolution Subnetwork hoàn chỉnh
    Kết hợp Transformation Module và Recovery Module
    """
    def __init__(self, in_channels=3, encoding_channels=[64, 128, 256], 
                 decoding_channels=[128, 64, 32], out_channels=3):
        """
        Args:
            in_channels: Số kênh đầu vào
            encoding_channels: Số kênh cho các lớp Conv
            decoding_channels: Số kênh cho các lớp Deconv
            out_channels: Số kênh đầu ra
        """
        super(FeatureEvolutionSubnetwork, self).__init__()
        
        # Transformation Module (Encoder)
        self.transformation = TransformationModule(
            in_channels=in_channels,
            hidden_channels=encoding_channels
        )
        
        # Recovery Module (Decoder)
        self.recovery = RecoveryModule(
            encoded_channels=encoding_channels[-1],
            hidden_channels=decoding_channels,
            out_channels=out_channels
        )
        
    def forward(self, x):
        """
        Forward pass qua toàn bộ subnetwork
        
        Args:
            x: Input tensor [batch, in_channels, H, W]
            
        Returns:
            output: Reconstructed output [batch, out_channels, H, W]
            encoding_features: Features từ Transformation Module
            decoding_features: Features từ Recovery Module
        """
        # Encoding phase
        encoding_features, encoded = self.transformation(x)
        
        # Decoding phase
        decoding_features, output = self.recovery(encoded)
        
        return output, encoding_features, decoding_features
    
    def encode(self, x):
        """Chỉ thực hiện encoding"""
        _, encoded = self.transformation(x)
        return encoded
    
    def decode(self, encoded):
        """Chỉ thực hiện decoding"""
        _, output = self.recovery(encoded)
        return output
