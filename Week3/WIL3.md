# Image Classification

2023.09.27 3주차

## 1. Why Computer Vision?

Computer Vision ↔ Computer Graphics

**Computer Graphics ( = Rendering )**

- 입력한 자료로 영상을 만드는 것 (데이터→이미지/영상)

**Computer Vision ( = Inverse Rendering )**

- 영상으로 의미있는 자료를 만들어내는 것 (이미지/영상 → 데이터)
- intelligence를 위해 인간의 지각 능력 중, 대부분의 정보를 처리하는 visual perception을 만듦

## 2. Image Classification

**Classifier**

- image를 해당하는 label에 매칭시키는 분류기
- 각 사물에 대한 정보를 훈련시켜 새로운 이미지에 적용

**Semantic Gap**

- 인간은 직관적으로 여러 환경에서도 사물을 구별할 수 있지만, 컴퓨터는 구별할 수 없음

→ 각도, 밝기, 크기, 포즈 등의 변화에 강한 알고리즘 필요

## 3. KNN (K-Nearest Neighbor)

- 비슷한 특성을 가진 데이터는 비슷한 범주에 속하는 경향이 있다는 가정
- 모든 훈련 데이터 셋을 기억하게 하여 주변의 가까운 **( *유클리드 거리 )** K개의 데이터를 보고 데이터가 속할 그룹이라고 판단하는 알고리즘
- K가 1, 즉 단순히 가까운 점 하나만으로 판단하면 오류 가능성 높음
- 항상 분류가 가능하도록 K는 홀수가 좋으며, 일반적으로 총 데이터 수의 제곱근 값을 사용

<aside>
💡 **Distance Metric**

**- Manhattan Distance ( L1 Distance )**

![Untitled](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled.png)

: RGB 픽셀 간 차이의 절댓값으로 측정, 경계선이 x,y축과 비슷하게 따라감

**- Euclidean Distance ( L2 Distance )**

![Untitled](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled%201.png)

: 경계선이 부드러움

</aside>

**KNN은 실제 image classification에서는 사용하지 않음** 

이유 1) 시간이 오래 걸림

이유 2) 이미지 간의 지각적 유사성 거리는 픽셀 간 거리와 같지 않음

이유 3) 차원의 저주, 즉, 차원이 커질수록 필요한 데이터 수가 늘어남

## 4. Linear Classification

**Image Captioning**

- image를 input으로 하고 해당 사진에 대한 설명을 ouput으로 함
- 컨볼루션 신경망(CNN) + 언어를 아는 순환 신경망(RNN)

 - 아래는 모두 CIFAR-10 Dataset 토대 -

<aside>
💡 **CIFAR-10 Dataset**
5000 training images
10000 test images
10 classes
image = 32x32x3

</aside>

Parametric Approach (모든 데이터를 저장하지 않고 파라미터값만 저장) 

- > **Linear Classification**

<aside>
✏️ f(x,W) = Wx + b

</aside>

- 이때 x는 입력 image (픽셀을 열벡터로 stretch), W는 가중치, b는 편향값

![출처 : [https://www.youtube.com/watch?v=OoUX-nOEjG0](https://www.youtube.com/watch?v=OoUX-nOEjG0)](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled%202.png)

출처 : [https://www.youtube.com/watch?v=OoUX-nOEjG0](https://www.youtube.com/watch?v=OoUX-nOEjG0)

- ‘f(x,W)로 가장 높은 점수를 받은 클래스에 속할 가능성이 높음’
- 그러나, 하나의 클래스를 하나의 템플릿으로만 학습하기에 (말머리가 두 개가 되는 등의) 문제 발생
    
    → 다양한 함수들을 블록처럼 복잡하게 쌓아야함
    

## 5. Loss Function

: classifier가 얼마나 잘 작동하는가, 모델의 출력값과 원하는 출력값의 오차

Loss Function = Object Function = Cost Function

## 6. Multiclass SVM Loss

![Untitled](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled%203.png)

![Untitled](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled%204.png)

정답 카테고리 yi를 제외한 나머지 카테고리 y의 합을 구하고, 정답 카테고리의 스코어와 틀린 카테고리의 스코어를 비교
→ 두 점수의 격차가 safety margin (여기서는 1)이상으로 정답 카테고리 점수가 더 높으면 loss가 0 (Li)

→ 정답이 아닌 모든 카테고리의 값들을 합친 값이 최종 Loss

→ 전체 트레이닝 셋에서 Loss들의 평균

## 7. Regularization : Beyond Training Error

![왼쪽 항은 전체 Data Loss, 오른쪽 항은 일반화 성능을 높이기 위한 Regularization](Image%20Classification%20ed951d2ffc434d46b1e9344aee25cfdd/Untitled%205.png)

왼쪽 항은 전체 Data Loss, 오른쪽 항은 일반화 성능을 높이기 위한 Regularization

**L2 Regularization**

- 매끄러운 그래프를 원할 때 사용
- 특정 요소만의 의존보다는 모든 요소의 전체적인 영향을 원하는 정규화

**L1 Regularization**

- 분류기가 복잡하다고 느껴지는 경우 사용
- 가중치에 0이 많도록 하여 보다 더 단순한 식