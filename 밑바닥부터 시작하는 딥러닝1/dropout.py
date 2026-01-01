import numpy as np
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    # 초기 구현 방식: 학습 때 평균 출력 ↓, 추론 때 기대값 맞추기 위해 스케일링
    # def forward(self, x, train_flg=True):
    #     if train_flg:
    #         self.mask = np.random.rand(*x.shape) > self.dropout_ratio
    #         return x * self.mask
    #     else:
    #         return x * (1.0 - self.dropout_ratio) # 추론 시 각 뉴런의 출력에 훈련 때 삭제 안 한 비율을 곱하여 출력
    
    # 학습 시 스케일링 (요즘 표준): PyTorch / TensorFlow
    def forward(self, x, train_flg=True):
        if train_flg:
            mask = np.random.rand(*x.shape) > self.dropout_ratio
            y = x * mask / (1.0 - self.dropout_ratio)
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask