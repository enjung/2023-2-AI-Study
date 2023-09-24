## WIL1
# Deep Learning Basic

2023.09.13 1주차

## AI, ML, DL의 차이와 관계

AI(Artificial Inteligence) : 인간의 능력을 인공적으로 구현

ML(Machine Learning) : **데이터**로부터 규칙을 학습하는 AI의 하위 분야

DL(Deep Learning) : **Neural network**를 기반으로 한 ML의 하위분야

![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled.png)

## Deep Learning Component

### 1) Data

- 다루는 Task에 dependent (Classification, Semantic Segmentation, Object Detection, Pose Estimation)

### 2) Model

- input에서 feature를 뽑고 우리가 원하는 output으로 만드는 프로그램

### 3) Loss function

- 학습 중 알고리즘이 얼마나 잘못 예측하는가에 대한 지표
- 알고리즘이 예측한 값과 실제 정답의 차이를 비교하여 학습
- 다루는 Task에 dependent (MSE, Cross Entropy, MLE, …)

### 4) Optimization and Regularization

Optimization

- Gradient Descent Method(경사하강법)
    - loss function을 빠르고 정확하게 줄이기 위한 최적화 기법

Regularization

- 학습을 방해하여 일반화 성능 ↑ (여러 학습 데이터에서도 잘 동작하도록 )

## Neural Network

: **Function Approximators** that stack affine transformations followed by **nonlinear** transformations

## Nonlinear Function

- Activation function(활성함수)로 비선형함수를 사용함

## Multi-Layer Perceptron

![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%201.png)

## Generalization

- 일반화 성능
- Generalization Gap = | Test error - Training error |

![학습을 반복할수록 training error는 감소하지만 0이 됐다고 해도 최적의 값은 아님(일정 구간이후부터는 test error가 증가함)](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%202.png)

학습을 반복할수록 training error는 감소하지만 0이 됐다고 해도 최적의 값은 아님(일정 구간이후부터는 test error가 증가함)

- **Under-fitting** : 학습데이터에서조차 잘 동작하지 않는 경우
- **Over-fitting(과적합)** : 학습데이터에서는 잘 동작하지만 테스트데이터에서는 잘 동작하지 않는 경우

![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%203.png)

- **Cross Validation (교차검증)**
    - train data중 valid data를 추출해 학습 확인 지표로 사용
- **Ensemble** - 여러 분류 모델을 조합해 더 나은 성능을 유도
    - **Bagging** : data set을 subset으로 나누어 학습 후 각각의 voting이나 averaging을 구함 (병렬 학습)
    - **Boosting** : 학습이 제대로 되지 않은 데이터들을 모아 새로운 간단한 모델로 재학습 (순차적 학습)
    
    ![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%204.png)
    
- **Regularization** - 학습을 방해
    - **early stopping** - over-fitting인 경우
    - **parameter norm penalty**
    - **data argumentation** - 이미지 회전/밝기조절/크롭/label수정 등 한정된 data로 많은 학습 가능
    - **noise robustness** - 이상치나 노이즈가 들어와도 크게 흔들리지 않음
    - **dropout** - 임의의 노드를 일정 확률로 drop함 (학습에 참여시키지 않음)
    - **label smoothing** - 모델이 너무 확신을 가지지 않도록(0과 1이 아니라) 도와주어 과적합을 줄임

## Convolutional Neural Networks (합성곱 신경망)

Fully connected multi layered Neural Networkd(FNN)의 문제점

- 인접 픽셀간의 상관관계가 무시되어 이미지를 벡터화하는 과정에서 정보손실

→ CNN으로 해결

**이미지의 공간정보를 유지하며 학습** vs 이미지의 공간정보를 유지하지 않으며 학습

![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%205.png)

<aside>
👀 {Conv → Activation funcion → pooling} x n → Fully connected layer

</aside>

Convolution 계산 방법

- input값(I)과 필터 역할의 행렬(K) 두 개 필요

![Untitled](Deep%20Learning%20Basic%20875396a2b4f54f888e3f8204ee52cccf/Untitled%206.png)

Pooling 계산 방법

- 파라미터 수와 연산량을 줄이기위한 다운사이징
- max-pooling, average-pooling

## 1x1 Convolution

- depth 차원 변경 가능 → neural network를 깊게 쌓을 수 있음

## Modern CNN

1) AlexNet

- 두 개의 네트워크
- 11x11 filter
- 5개의 convolution layer, 3개의 dense layer

2) VGGNet

- 3x3 convolution filer

3) GoogLeNet

- 1x1 convolution

4) ResNet

- 사람의 능력을 뛰어넘은 첫 번째 모델
- gradient vanishing problem을 skip connection을 도입해 해결