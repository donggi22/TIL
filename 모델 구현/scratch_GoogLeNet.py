import torch
import torch.nn as nn

class Inception_Module(nn.Module):
    def __init__(self, in_channels, filters_num_array):
        super().__init__()
        
        br0_filters = filters_num_array[0]
        br1_filters = filters_num_array[1]
        br2_filters = filters_num_array[2]
        br3_filters = filters_num_array[3]

        self.br0_conv = self._conv_bn_relu(in_channels=in_channels, out_channels=br0_filters, kernel_size=1)

        self.br1_conv1 = self._conv_bn_relu(in_channels=in_channels, out_channels=br1_filters[0], kernel_size=1)
        self.br1_conv2 = self._conv_bn_relu(in_channels=br1_filters[0], out_channels=br1_filters[1], kernel_size=3, padding=1)

        self.br2_conv1 = self._conv_bn_relu(in_channels=in_channels, out_channels=br2_filters[0], kernel_size=1)
        self.br2_conv2 = self._conv_bn_relu(in_channels=br2_filters[0], out_channels=br2_filters[1], kernel_size=3, padding=1)

        self.br3_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.br3_conv = self._conv_bn_relu(in_channels=in_channels, out_channels=br3_filters, kernel_size=1)

    # 헬퍼함수
    def _conv_bn_relu(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """레이어를 반환 (호출 X)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )
    # 호출은 self.conv(x) 처럼 forward 호출을 말함.

    # nn.Conv2d(...): 인스턴스(레이어) 생성
    # self.conv = nn.Conv2d(...): nn.Module의 서브모듈(레이어)로 등록
    # self.conv(x): forward 호출
    # return nn.Sequential(...): 모듈 반환
    # nn.Sequential(...)(x): 생성 + 즉시 호출

    def forward(self, x):
        br0 = self.br0_conv(x)

        br1 = self.br1_conv1(x)
        br1 = self.br1_conv2(br1)

        br2 = self.br2_conv2(x)
        br2 = self.br2_conv2(br2)

        br3 = self.br3_pool(x)
        br3 = self.br3_conv(br3)

        # Concatenate
        out = torch.cat([br0, br1, br2, br3], dim=1)
        return out
    
# 보조 분류기
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(128, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, aux_logits=True):
        super().__init__()
        
        self.aux_logits = aux_logits  # 보조 분류기 사용 여부 (원본 논문처럼 True가 기본)

        # Stem
        self.conv1 = self._conv_bn_relu(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = self._conv_bn_relu(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = self._conv_bn_relu(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Inception 모듈
        self.inception3a = Inception_Module(in_channels=192, filters_num_array=(64, (96, 128), (16, 32), 32))
        self.inception3b = Inception_Module(in_channels=256, filters_num_array=(128, (128, 192), (32, 96), 64))
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=True)

        self.inception4a = Inception_Module(in_channels=480, filters_num_array=(192, (96, 208), (16, 48), 64))
        self.inception4b = Inception_Module(in_channels=512, filters_num_array=(160, (112, 224), (24, 64), 64))
        self.inception4c = Inception_Module(in_channels=512, filters_num_array=(128, (128, 256), (24, 64), 64))
        self.inception4d = Inception_Module(in_channels=512, filters_num_array=(112, (144, 288), (32, 64), 64))
        self.inception4e = Inception_Module(in_channels=512, filters_num_array=(256, (160, 320), (32, 128), 128))
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception_Module(in_channels=832, filters_num_array=(256, (160, 320), (32, 128), 128))
        self.inception5b = Inception_Module(in_channels=832, filters_num_array=(384, (192, 384), (48, 128), 128))
        
        # 보조 분류기들 (auxiliary classifiers)
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)  # inception4a 출력 후
            self.aux2 = AuxiliaryClassifier(528, num_classes)  # inception4d 출력 후

        # 메인 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def _conv_bn_relu(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """레이어를 반환 (호출 X)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        # model.train()  # model.training = True
        # model.eval()   # model.training = False

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


#-------------------------------------------------------------------------------------------------------------------

# Inception_Module 내부에 _conv_bn_relu 메서드를 만들지 않고 클래스로 합성곱 블록을 만들어서 재사용 가능하게도 할 수 있음!!

class ConvBnReLU(nn.Module):
    """재사용 가능한 Conv-BN-ReLU 블록"""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 위와 아래는 같은 역할 
class ConvBnReLU2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    

class InceptionModule(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        
        # Branch 0: 1×1
        self.branch0 = ConvBnReLU(in_channels, filters[0], 1)
        
        # Branch 1: 1×1 → 3×3
        self.branch1 = nn.Sequential(
            ConvBnReLU(in_channels, filters[1][0], 1),
            ConvBnReLU(filters[1][0], filters[1][1], 3, padding=1)
        )
        
        # Branch 2: 1×1 → 5×5
        self.branch2 = nn.Sequential(
            ConvBnReLU(in_channels, filters[2][0], 1),
            ConvBnReLU(filters[2][0], filters[2][1], 5, padding=2)
        )
        
        # Branch 3: pool → 1×1
        self.branch3_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch3_conv = ConvBnReLU(in_channels, filters[3], 1)
    
    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3_conv(self.branch3_pool(x))
        
        return torch.cat([b0, b1, b2, b3], dim=1)


# 사용
model = InceptionModule(192, [64, [96, 128], [16, 32], 32])
x = torch.randn(1, 192, 28, 28)
out = model(x)
print(out.shape)  # [1, 256, 28, 28]