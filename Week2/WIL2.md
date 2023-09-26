# Intro CV & Pytorch

2023.09.20 2주차

## 1. Computer Vision Applications

## 2. Semantic Segmentation

### segmantation

- 입력 이미지를 픽셀 수준에서 분석하여 각 픽셀에 클래스 레이블을 할당
- **semantic segentation**( 같은 클래스는 구분x ) VS **instance segmentation** (같은 클래스여도 instance별로 구분)

**CNN** - 네트워크 뒷단에 **fully connected layer**를 붙여 image classification을 수행

**FCN** - 네트워크 뒷단에 **CNN**를 붙여 image classification을 수행 

즉, FCN은 FC Layer의 이미지 위치 정보가 사라지고 input image의 사이즈가 고정되는 한계점을 극복)

FCN의 upsampling

- deconvolution (Conv의 역연산 )
- interpolation ( 주어진 값들 사이의 값을 linear하게 추정)

## 3. Object Detection

: bounding box로 객체의 위치를 찾는 Tast

**R-CNN (Regions with CNN features)**

- 시간이 오래 걸림 (2천개의 region proposal에 대해 각각 CNN 연산)
    - Selective Search 알고리즘으로 임의의 bounding box 설정 (랜덤하게 작게 많이 생성한 후 그룹핑 알고리즘으로 merge하여 이를 바탕으로 ROI를 제안하는 region proposal)
    - → 추출한 region proposal을 동일 input size로 만들어주기 위해 warp
    - → 2000개의 warped image를 각각 CNN모델에 적용
    - → 추출된 feature를 SVM을 통해 classification

**SPPNet**

- 2k CNN적용이 필요한 R-CNN에 비해 SPPNet은 한 번의 CNN연산만 진행
- 2000개 영역에 해당하는 feature값만 뜯어와 속도  ↑

**Fast R-CNN**

- 마찬가지로 한 번의 CNN연산
- 구조를 더 단순화하여 연산시간 줄이고 성능을 높임
    - binary SVM을 사용하지 않고 FC layer와 softmax를 활용해 클래스 구별
    - CNN으로 추출한 feature vector를 bbox regressor의 입력으로 사용 → 하나의 입력으로 feature 추출, classification, bbox regression을 한 모델에서 연산 가능
- ROI Pooling : 패치들을 고정된 feature값으로 만들기 위한 작업, fully connected layer에 전달
- 객체 후보 영역에 대해 추출하는 multi-pipeline구조

**Faster R-CNN**

- Fast R-CNN + RPN(Region Proposal Network)
- RPN
    - bottleneck이었던 Selective Search알고리즘 대신 객체 후보 영역을 추정하는 네트워크 RPN을 사용

**YOLO(You Look Only Once)**

- region proposal 단계가 x
- **NMS**(Non-Maximum Suppression) 알고리즘
    - 동일 그리드 영역으로 나누기
    - → bbox와 박스에 대한 신뢰도 점수를 예측, classification
    - → 신뢰도가 높은 굵은 박스들만 남김
- 신뢰도 점수 예측과 classification을 **동시 진행**
- 빠르고 간단하지만 작은 객체 인식률은 떨어짐

## 4. Introduction to Pytorch

Framework

: 응용 프로그램을 개발하기 위한 여러 라이브러리나 모듈 등을 효율적으로 사용할 수 있도록 하나로 묶어 놓은 패키지

DL Framework - Tensorflow, **Pytorch**, keras, JAX

## 5. Pytorch Basics

**Tensor**

- 다차원 배열을 표현하는 자료구조
- umPy의 ndarray와 유사
- list나 ndarray를 사용해 생성 가능
- GPU에 올려서 사용 가능

x.**view[ ]** : shallow copy (memory 주소 copy) (contiguous)

x.**reshape[ ]** : deep copy (value copy)

```python
tensor_ex = torch.rand(size=(4,4,3))
tensor_ex.view[-1,3] >> 16x3
tensor_ex.view[-1,2,3] >> 8x2x3
tensor_ex.reshape[-1,3] >> 16x3
tensor_ex.reshape[-1,2,3] >> 8x2x3
```

x.**squeeze**( ) : 특정 차원을 제거(디폴트는 1인 차원)
x.**unsqueeze**( ) : 특정 위치에 1인 차원을 추가

```python
x = torch.rand(1,7,46,46)
print(x.shape)  # torch.Size([1, 7, 46, 46])

x = x.squeeze(dim=1)
print(x.shape)  # torch.Size([7, 46, 46])

x = torch.rand(7,46,46)
print(x.shape)  # torch.Size([7, 46, 46])

x = x.unsqueeze(dim=2)
print(x.shape)  # torch.Size([7, 46, 1, 46])
```

dot : 행렬곱을 지원하지 x

mm : 행렬곱만 지원(broadcasting 지원 x)

matmul : 행렬곱 지원, broadcasting 지원 

nn.functional

- softmax : 입력받은 값을 출력으로 0~1 사이 값으로 정규화 (출력 값들의 총 합은 1)
- one-hot encoding : 해당하는 class만 1, 나머지는 0

autograd : pytorch의 자동 미분 엔진

- backward함수

## 6. PyTorch 프로젝트 구조 이해하기

## 7. AutoGrad & Optimizer

**nn.Module**

- Pytorch의 모든 Neural Network의 Base Class

**nn.Parameter**

- 자동미분되는 torch.Tensor (required_grad = True)
- 파라미터로 지정하고 싶은 텐서일 때 사용
- 대부분의 layer에는 초기화된 무작위 가중치값이 정해짐 (파라미터 직접 추가할 일 거의 x)

**Backward**

- 파라미터들의 미분을 수행하는 함수
- forward 결과와 실제 값의 차이를 현재 가중치에 대해 미분한 편미분값을, 가중치에서 뺀 값으로 파라미터 업데이트

## 8. Pytorch Dataset

**Dataset 클래스**

- 데이터셋(데이터 sample 정제, label을 저장)을 나타내는 추상클래스
- 데이터 입력 형태를 정의, 데이터 입력 방식을 표준화

**DataLoader 클래스**

- data의 batch(뮦음)를 생성
- 모델에 데이터를 넣기 전, 데이터를 수월하게 나눌 수 있도록 도움

## 9. Model Save

**model.state_dict()**

- 모델 전체를 저장하는 게 아니라 학습 가능한 매개변수가 담겨있는 딕셔너리만 저장
- 전체 저장보다 용량이 작음

## 10. Transfer Learning

: pretrained model(남이 학습시킨 모델)로 내 데이터를 다시 학습 시키는 것

- 내가 처음부터 만든 모델에 비해 학습 데이터가 많다는 장점
- 마지막 layer만 내 데이터로 재학습