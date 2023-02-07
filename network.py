from torchvision import models
import torch.nn as nn
import torch

# ResNet Feature Extractor
class ResNetFeatureExtractor(nn.Module):
	def __init__(self, backbone = 'resnet50', pretrained = True, retrain = False, **kwargs):
		super(ResNetFeatureExtractor, self).__init__()
		self.pretrained = pretrained
		self.backbone = backbone
		if self.backbone == 'resnet50':
			self.resnet_backbone = models.segmentation.fcn_resnet50(pretrained=pretrained).backbone
		elif self.backbone == 'resnet101':
			self.resnet_backbone = models.segmentation.fcn_resnet101(pretrained=pretrained).backbone
		else:
			raise ValueError('Backbone not supported')
		if retrain == False:            
			self.resnet_backbone.eval()
			for param in self.resnet_backbone.parameters():
				param.requires_grad = False
	def forward(self, x):
		x = self.resnet_backbone(x)
		return x

# Inherit backbone from ResNetFeatureExtractor for segmentation
class SegmentationNetwork(nn.Module):
	def __init__(self, num_classes, p=0.5, backbone = 'resnet50', pretrained = True, **kwargs):
		super(SegmentationNetwork, self).__init__()
		self.backbone = ResNetFeatureExtractor(backbone, pretrained, retrain = True)
		self.decoder = nn.Sequential(
		# Keep the size of the feature map same
		nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=True),
		nn.BatchNorm2d(1024),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 28x28 to 56x56
		nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = True),
		nn.BatchNorm2d(512),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 56x56 to 112x112
		nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = True),
		nn.BatchNorm2d(256),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 112x112 to 224x224
		nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = True),
		nn.BatchNorm2d(128),
		nn.ReLU(inplace=True)
		)
		self.dropout = nn.Dropout(p=p)
		self.classifier = nn.Conv2d(128, num_classes*2, kernel_size=1)
	def forward(self, x):
		if isinstance(x, dict):
			x_rgb = x['rgb']
			x_rgb = self.backbone(x_rgb)
			x = self.decoder(x_rgb['out'])
			x = self.dropout(x)
			x = self.classifier(x)
		else:
			x = self.backbone(x)
			x = self.decoder(x['out'])
			x = self.dropout(x)
			x = self.classifier(x)
		return x

# Segmentation Network with hierarchical feature fusion
class SegmentationNetwork_HFF(nn.Module):
	def __init__(self, num_classes, dropout=0.5, backbone = 'resnet50', pretrained = True, **kwargs):
		super(SegmentationNetwork_HFF, self).__init__()
		self.resnet_backbone = ResNetFeatureExtractor(backbone, pretrained)
		self.classifier = nn.Sequential(
		nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = False),
		nn.BatchNorm2d(1024),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 28x28 to 56x56
		nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = False),
		nn.BatchNorm2d(512),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 56x56 to 112x112
		nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = False),
		nn.BatchNorm2d(256),
		nn.ReLU(inplace=True),
		# transpose conv to increase resolution from 112x112 to 224x224
		nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1, bias = False),
		nn.BatchNorm2d(128),
		nn.ReLU(inplace=True),
		# classifier with dropout
		nn.Dropout2d(p=dropout),   
		nn.Conv2d(256, num_classes*2, kernel_size=1),
		)
	def forward(self, x):
		# Return intermediate features from ResNet backbone
		x = self.resnet_backbone(x)
		# Concatenate intermediate features from ResNet backbone
		x = torch.cat((x['out'], x['out2'], x['out3'], x['out4']), dim=1)
		# Pass concatenated features through classifier
		x = self.classifier(x)
		return x

if __name__ == '__main__':
	model = SegmentationNetwork_HFF(2)
	x = torch.randn(1, 3, 224,224)
	y = model(x)
	print(y.shape)