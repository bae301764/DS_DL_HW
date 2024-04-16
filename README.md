# DS_DL_HW
서울과학기술대학교_데이터사이언스학과_인공신경망과 딥러닝 과제

## 1. LeNet-5과 CustomMLP의 모델 구조와 파라미터 수에 대한 설명
파라미터 수는 torchsummary라는 패키지에서 summary 함수를 이용하여 계산하였습니다.

- LeNet-5의 모델 구조
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/LeNet%20%EA%B5%AC%EC%A1%B0.png">

|layer|number of param|구성|
|------|---|---|
|conv2d-1|156|kernel size:5x5, in_channels=1, out_channels:6, bias:6  => 1x25x6+6|
|conv2d-4|2,416|kernel size:5x5, in_channels=6, out_channels:16, bias:16  => 6x25x16+16|
|conv2d-7|30,840|kernel size:4x4, in_channels=16, out_channels:120, bias:120  => 16x16x120+120|
|Linear-9|10,164|input node=120, output node:84, bias:84  => 120x84+84|
|Linear-11|850|input node=84, output node:10, bias:10  => 84x10+10|



- CustomMLP의 모델 구조
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/customMLP%20%EA%B5%AC%EC%A1%B0.png">

|layer|number of param|구성|
|------|---|---|
|Linear-1|39,250|input node=28x28, output node:50, bias:50  => 28x28x50+50|
|Linear-3|3,060|input node=50, output node:60, bias:60  => 50x60+60|
|Linear-5|1,830|input node=60, output node:30, bias:30  => 60x30+30|
|Linear-7|310|input node=30, output node:10, bias:10  => 30x10+10|


## 2. LeNet-5와 CustomMLP 성능 비교 & LeNet-5에 Regularization 추가한 후 비교
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/loss%20and%20accuracy%20plot.png">

### LeNet-5 와 CustomMLP
- LeNet-5(0.9778)가 CustomMLP(0.9582)에 비해 성능이 우수함

### LeNet-5 와 Regularization이 추가된 모델(LeNet Advance)
- LeNet-5의 첫 번째와 두 번째 Convolution 단계에 Batch Normalization를 추가
- L2 regularization인 weight decay를 추가
- Regularization을 사용함으로써 일반화 성능(0.9809)이 향상되었음을 확인


## 3. hyperparameter 세부사항
activation funtion : tanh()\
n_epochs : 30\
weight decay : 1e-3\
batch size = 512
