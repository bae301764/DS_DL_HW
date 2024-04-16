# DS_DL_HW
서울과학기술대학교_데이터사이언스학과_인공신경망과 딥러닝 과제

## 1. LeNet-5과 CustomMLP의 모델 구조와 파라미터 수에 대한 설명
파라미터 수는 torchsummary라는 패키지에서 summary 함수를 이용하여 계산하였습니다.
- LeNet-5의 모델 구조
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/LeNet%20%EA%B5%AC%EC%A1%B0.png">
- CustomMLP의 모델 구조
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/customMLP%20%EA%B5%AC%EC%A1%B0.png">

## 2. LeNet-5와 CustomMLP 성능 비교 & LeNet-5에 Regularization 추가한 후 비교
<img src="https://github.com/bae301764/DS_DL_HW/blob/main/loss%20and%20accuracy%20plot.png">
### LeNet-5 와 CustomMLP
- LeNet-5가 CustomMLP에 비해 성능이 우수함
### LeNet-5 와 Regularization이 추가된 모델(LeNet Advance)
- LeNet-5의 첫 번째와 두 번째 Convolution 단계에 Batch Normalization를 추가
- L2 regularization인 weight decay를 추가
- Regularization을 사용함으로써 일반화 성능이 향상되었음을 확인

## 3. hyperparameter 세부사항
activation funtion : tanh()
n_epochs : 30
weight decay : 1e-3
batch size = 512
