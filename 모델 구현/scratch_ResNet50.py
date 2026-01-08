import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4 # conv3층의 out_channels를 이전층 out_channels의 몇 배수로 할지
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,  # self.expansion은 클래스 멤버
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample # skip 연결용
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x) # [64,H,W] → [256,H/2,W/2], stride로 다음 층 이미지 크기 조절
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        
        # Downsample if stride != 1 or channels don't match
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# 사용 예시
if __name__ == "__main__":
    model = ResNet50(num_classes=3)  # COVID-19 3-class classification
    
    # 모델 구조 확인
    print(model)
    
    # Input test
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(output)


#    Input (3 × H × W)
# │
# ├─ Conv1: 7×7, 64 filters, stride 2
# ├─ BatchNorm
# ├─ ReLU
# ├─ MaxPool: 3×3, stride 2
# │
# ├─ Layer1 (Conv2_x): 3 Bottleneck Blocks
# │   └─ [1×1, 64] → [3×3, 64] → [1×1, 256] × 3
# │
# ├─ Layer2 (Conv3_x): 4 Bottleneck Blocks
# │   └─ [1×1, 128] → [3×3, 128] → [1×1, 512] × 4
# │   ※ 첫 블록 stride = 2 (다운샘플링)
# │
# ├─ Layer3 (Conv4_x): 6 Bottleneck Blocks
# │   └─ [1×1, 256] → [3×3, 256] → [1×1, 1024] × 6
# │   ※ 첫 블록 stride = 2
# │
# ├─ Layer4 (Conv5_x): 3 Bottleneck Blocks
# │   └─ [1×1, 512] → [3×3, 512] → [1×1, 2048] × 3
# │   ※ 첫 블록 stride = 2
# │
# ├─ AdaptiveAvgPool2d(output=1×1)
# ├─ Flatten
# └─ Fully Connected: Linear(2048 → 4)