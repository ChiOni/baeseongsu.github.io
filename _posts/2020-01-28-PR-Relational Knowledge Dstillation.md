---
layout: post
title:  "Paper Review : Relational Knowledge Distillation"
date:   2020-01-28 22:22
categories: [Deep Learning, Knowledge Distillation]
use_math: true
---

안녕하세요. 배성수입니다.

오늘 리뷰할 논문은 CVPR 2019에 Accepted Paper인 **Relational Knowledge Distillation** 입니다.

최근 딥러닝 스터디를 하며 Knowledge Distillation(KD) 관련 논문 2편을 읽었습니다. NIPS 2014 Deep Learning Workshop에서 발표한 **Distilling the Knowledge in a Neural Network** , ICML 2019에 Accepted Paper인 **Zero-Shot Knowledge Distillation in Deep Networks** 입니다.

극히 주관적인 저의 생각입니다만, 앞선 2편의 KD 논문에 비해 오늘 리뷰할 논문이 좀 더 명확하고 깔끔했습니다. 특히 핵심 아이디어가 뚜렷하고, 뒷받침하는 실험들의 세팅이 탄탄하다는 느낌을 받았습니다. 저자는 POSTECH의 Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho 입니다. 재밌게 잘 읽었습니다. 감사합니다.

<br/>

---

<br/>

# <center> Relational Knowledge Distillation, CVPR 2019 </center>

<br/>

## 1. Introduction

최근 Computer Vision이나 Artificial Intelligence 연구에선 많은 연상량과 메모리를 필요로 하는 모델들이 자주 등장합니다. 이러한 물리적 부담을 줄이기 위한 방법 중 하나로 모델의 지식(knowledge)을 전달(transfer)하는 방법이 있습니다. 이러한 Knowledge Transfer에 있어서, 가장 핵심이 되는 2가지 질문이 있습니다. 바로 "학습된 모델에 들어있는 지식은 무엇으로 구성되어 있는가?"와 "그 지식을 다른 모델로 어떻게 전달할 것인가?"입니다.

예를 들어, Hinton 교수님의 KD 방법에서 지식이란, 입력으로부터 출력까지 학습된 매핑함수를 의미합니다. 또한, Teacher 모델의 Soft Targets을 이용해 지식을 전달하는 방식을 취하고 있습니다. 이처럼 두 가지 질문으로부터 KD 연구는 진행된다고 해도 과언이 아닙니다.

특히, 이 논문은 언어적 구조주의(linguistic structuralism) 관점에서 KD가 중요하게 생각해야할 점들에 대해 설명하고 있습니다. 언어도 일종의 기호학 체계로 볼 수 있고, 언어에 녹아든 기호의 의미는 큰 체계 속에서 다른 기호들과 어떤 관계를 맺고 있는지가 중요하다는 뜻입니다. (자세한 내용은 논문과 소쉬르의 구조주의 언어학을 참조해주세요.)

그렇다면, 앞서 말한 2가지 질문에 대해 이 논문은 어떻게 답할 수 있을까요? 먼저, "지식이라는 것은 (학습된 상황일 때) 개별적인 representation보다 그들의 관계에 의해 더 잘 표현된다"라는 것입니다. 그에 따라, "개별적인 output보다 output들의 구조적 관계를 전달하는 방식으로 지식을 전달하는 방식으로 접근할 것이다"라고 설명하고 있습니다. 이러한 점에서 기존 KD를 일반화할 수 있는 RKD라는 것이 탄생하게 되었고, 지식전달능력도 상당히 우수하다고 보였습니다. 결국, 지식은 관계 속에 녹아들어 있고, RKD는 이러한 지식을 전파하는데 있어 효과적인 방법이라고 설명합니다.

![](/assets/img/pr/rkd_1.png) 

<br/>

<br/>

## 2. Related Work

한 모델의 지식(Knowledge)을 다른 모델로 전달(Transfer)하는 연구는 꽤 오랫동안 해왔습니다. 처음으로, Breiman and Shang이 트리 기반의 model compression을 통해 지식을 전달하는 방법을 제안했다고 합니다. 그 이후로, 신경망 분야의 model compression이 등장했고, Hinton 교수님은 soft targets라는 컨셉을 이용하여 지식 증류(Knowledge Distillation)라는 네이밍을 탄생시켰습니다. 최근에는 HKD(Hinton's KD)를 이은 후속 연구들뿐만 아니라 기존 접근과 다른 방식의 연구가 진행되고 있으며, 지도학습을 넘어 준지도학습/비지도학습 영역에서의 KD, 태스크에 특화된 KD 등에 대한 연구가 진행되고 있습니다.

다양한 KD 연구흐름 속에서 Chen의 연구(Darkrank: Accelerating deep metric learning via cross sample similarities transfer)가 rank loss를 사용해 similarities를 transfer하는 metric learning 기반의 KD라는 점에서 이 연구와 유사성이 어느정도 있습니다. 그러나, Chen의 연구는 metric learning에만 제한되어 있고, 본 연구는 다양한 테스크에 적용가능한 general framework라는 차이점이 있습니다. 게다가, metric learning task에서 Chen의 KD방법보다 성능이 더 좋았다고 합니다.

<br/>

<br/>

## 3. Our Approach

먼저, 보편적으로 사용해 온 지식증류(Knowledge Distillation)를 살펴본 뒤 논문에서 제안한 관계형 지식증류(RKD, Relational Knowledge Distillation)의 핵심 개념에 대해 살펴볼 것입니다. 또한 RKD에서 사용되는 손실함수로서, 간단하면서도 효과적인 두 가지 증류 손실함수(distillation losses)에 대해 소개하는 순서로 글을 작성했습니다.

<br/>

### 3.0 Notation

먼저, 논문에서 사용하는 Notation에 대해 알아보겠습니다.

(1) 주어진 Teacher model $T$ , Student model $S$ 이 일반적으로 Deep Nueral Network라고 형태라고 생각했을 때, 해당 모델의 mapping function을 각각 $f_T$ 와 $f_S$ 라고 표기합니다. $f$ 라는 함수는 신경망의 어떤 층이든 상관없이 그 층의 출력으로 정의될 수 있으나, 보통은 최종 출력을 의미할 때가 많습니다.

(2) 서로 다른 data examples의 $N$-튜플 형태를 $\chi^{N}$이라고 표기합니다. 예를 들면, $\chi^{2}$ 라면 $\\{ ( x_i, x_j ) \, \| \,  i \neq j  \\}$ 와 같은 distinct pair set, $\chi^3$ 라면 $\\{ (x_i, x_j, x_k) \, \| \,  i \neq j \neq  k \\}$ 와 같은 distinct triplet set로 볼 수 있습니다.

<br/>

### 3.1 Conventional KD

일반적인 KD 방법은 다음과 같은 목적함수를 갖습니다.

$$\mathcal{L}_{\text{IKD}} = \sum_{x_i \in \chi}{l( f_T(x_i), f_S(x_i) )}$$

즉, Teacher와 Student 모델로 나온 각각의 output mapping을 비슷하게 만들도록 학습합니다. 논문에서는, 이러한 종류의 KD들이 개별적인 Teacher 모델의 출력값을 Student 모델에게 전해준다는 점에서 **IKD(Individual Knowledge Distillation)**라고 명명할 수 있다고 말합니다.

<br/>

### 3.2 Relational KD

RKD의 목표는 Teacher 모델의 output representation에서 data examples의 mutual relations를 이용해 구조적 지식(Structural Knowledge)를 전달하는 것으로 볼 수 있습니다. 따라서 기존 KD가 지식을 전달하는 방식과 달리, RKD는 $n$-tuple 형태의 data examples에 대한 relational potential $\psi$ 를 계산하고, 그 포텐셜 값을 통해 지식을 전달합니다.

$$\mathcal{L}_{\text{RKD}} = \sum_{(x_{i},.., x_{j})\in\chi^{2}}{l_{\delta}{(\psi{(t_i,..,t_j)}, \psi{(s_i,..,s_j)})}}$$

$$ \text{where } t_i =  f_{T}(x_{i}), \; s_{i} = f_{S}(x_{i}), \; \psi : \text{relational potential function}$$ 







- - - measures a relational energy of the given n-tuple

RKD는 relational potential function을 사용해 나온 teacher 모델의 relational structure에 대해 같게 가지도록 student model을 학습시킴

potential 덕분에, high-order properties의 지식을 transfer할 수 있다.

- Thanks to the potential, it is able to transfer knowledge of high-order properties, which is invariant to lower- order properties, even regardless of difference in output di- mensions between the teacher and the student.

(4)식이 (1)식으로 줄일 수 있다는 관점에서 본다면, IKD의 일반화된 것이 RKD임

- RKD의 relation이 unary (N=1)일 때, 그리고 potential function $$\psi$$가 identity일 때, IKD임

RKD에서 relational potential function은 굉장히 중요함

- potential function에 따라 RKD의 효과와 효율성이 달려있음
- 예를 들어, higher-order potential은 higher-level structure을 캡쳐하는데 강력할진 몰라도 computationally expensive

- 그래서, 2개의 간단하지만 효과적인 potential functions를 제안하고, 이에 상응하는 RKD의 losses에 대해 제안함
  - 짝을 찟어 본 것 - distance-wise loss
  - 세 개를 하나의 쌍으로 관계로 생각 - angle-wise loss

<br/>

#### 3.2.1 Distance-wise distillation loss

$\psi_{\text{D}}$ 라는 거리 기반의 포텐셜 함수(distance-wise potential function)를 $\psi_{\text{D}} (t_{i},t_{j}) = \frac{1}{\mu}{\|\|t_{i}-t_{j}\|\|}_2$ 라고 정의합니다. 즉, 한 쌍을 이루는 두 개의 데이터 샘플이 신경망을 통해 output representation space에 놓여질 때, 그들간의 유클리디안 거리를 계산하는 함수라고 보시면 됩니다. 여기서 $\mu$ 는 거리함수의 normalization factor 입니다. 그렇다면, 이 $\mu$ 는 어떻게 정하는 것이 좋을까요?

<br/>

논문의 핵심 아이디어가 결국 관계성에 있기 때문에, 다른 쌍들과 비교하여 상대적 거리를 구할 수 있게 $\mu$ 를 선택하게 됩니다. 따라서 쌍으로 구성된 미니배치인 $\chi^{2}$ 에서 나온 각각의 페어 데이터의 평균 거리로 계산하게 됩니다.

$$  {\mu = \frac{1}{\| \chi^2 \|} \sum_{(x_i, x_j) \in \chi^{2} } {\| t_i-t_j \|_2}}  $$

만약 $\mu$ 와 같은 factor가 존재하지 않는다면, Teacher 모델의 dimension 일반적으로 더 크기 때문에 Teacher 모델과 Student 모델 사이의 거리 scale 차이가 발생하게 됩니다. 따라서 논문에서는 $\mu$ 를 사용하여 $\psi_{\text{D}}$ 라는 포텐셜 함수가 결국 distance-wise potentials를 잘 반영할 수 있도록 합니다. 실제로 $\mu$ 라는 factor로 인해 학습이 더 안정적이고 빠르게 수렴하는 것을 관찰했다고 합니다.

<br/>

앞에서 살펴본 $\psi_{\text{D}}$ 를 사용해 거리기반 증류 손실함수(Distance-wise distillation loss)는 다음과 같이 정의합니다.

$$\mathcal{L}_{\text{RKD-D}} = \sum_{(x_{i}, x_{j})\in\chi^{2}}{l_{\delta}{(\psi_{\text{D}}{(t_i,t_j)}, \psi_{\text{D}}{(s_i,s_j)})}}$$

<center> (여기서 $l_{\delta}$는 Huber loss를 의미, 외부에선 MAE, 이상치 덜 민감 - 내부에선 세밀하게 MSE) </center>
결국 이 손실함수는 모델의 output representation spaces 내 상대적 거리를 비슷하게 만들도록 해서, 쌍으로 이루어진 데이터가 있을 때 그들의 관계(relationships)들을 Student 모델로 전달(transfer)하는 역할을 하게 됩니다. Student의 output이 직접적으로 Teacher 모델의 output 값을 맞추도록 강요하는 것이 아니라, output이 놓여지는 공간의 거리구조에 초점을 맞추도록 한다는 것이 기존 KD와의 차이점이라고 할 수 있습니다.

<br/>

#### 3.2.2 Angle-wise distillation loss

앞에서 살펴본 $\psi_{\text{D}}$ 가 pair로 작동하는 방식이었다면, 하나의 차원이 더 늘어난 triplet은 어떤 방식으로 작동할까요? 세 쌍이 주어진 경우, output representation space에서 생기는 angle에 대한 metric을 생각해볼 수 있습니다. 따라서, angle-wise potential function $\psi_{\text{A}}$ 는 다음과 같이 정의할 수 있습니다.


$$ \psi_{\text{A}}{(t_i, t_j, t_k)} = cos \angle{t_{i}t_{j}t_{k}} = \langle \mathbf{e}^{ij}, \mathbf{e}^{kj} \rangle$$ 

$$\text{where } \mathbf{e}^{ij} = \frac{t_i-t_j}{\|t_i-t_j\|_2}, \mathbf{e}^{kj} = \frac{t_k-t_j}{\|t_k-t_j\|_2}.$$ 

<br/>

동일한 방식으로 각도 기반의 증류 손실함수(Angle-wise distillation loss)를 생각한다면, 다음과 같이 표기할 수 있습니다.

$$\mathcal{L}_{\text{RKD-A}} = \sum_{(x_i,x_j,x_k)\in\chi^{3}}{l_{\delta}{(\psi_{\text{A}}{(t_i,t_j,t_k)}, \psi_{\text{A}}{(s_i,s_j,s_k)})}}$$

기존의 distance-wise 보다 angle-wise가 더 higher-order property이기 때문에, 학습 과정에서 관계형 정보를 Student 모델에게 더욱 효과적이고 유연하게 전달할 수 있습니다. 실제 실험에서, angle-wise loss가 종종 더 빠르고 수렴하고, 좋은 성능을 보이는 것을 관찰했다고 합니다.

<br/>

#### 3.2.3 Training with RKD

학습과정에서 제안된 RKD 손실함수는 단독으로 사용할 수도 있고,  task에 특화된 손실함수와 함께 사용할 수도 있습니다. 따라서 전체적인 목적함수(objective)를 수식으로 표현하면 아래와 같은 형태가 됩니다.

$$\mathcal{L}_{\text{task}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}}$$

<br/>

추가적으로 이 논문에서 RKD에서 제안된 증류 손실함수들을 구할 때는, tuple sampling을 미니배치 속 표본에 대해 가능한 모든 조합으로 구성한다고 합니다. $\lambda$ 와 같은 balancing factor는 모델의 하이퍼파라미터로서 작동합니다.

<br/>

#### 3.2.4 Distillation target layer

RKD에서 distillation target function $f$ 는 이론적으로 어떤 레이어의 output mapping을 쓰든 상관이 없습니다. 하지만, distance/angle-wise losses는 Teacher 모델의 개별적인 출력값에 대한 지식을 전달해주지 않기 때문에 개별 output값들 자체가 중요한 곳에 혼자 덜렁 사용하는 것은 적절치 않습니다. 그런 경우에는 IKD 손실함수나 task-specific 손실함수를 사용하는 것이 필요합니다. 그 밖에 대부분의 다른 경우는 RKD가 적용가능하고 효과적인 성능을 보인다고 합니다.

<br/>

<br/>

## 4. Experiments

metric learning, classification, few-shot learning 이라는 3가지 태스크에 대해 실험을 진행했습니다. 기존의 RKD를 사용한 손실함수에 따라 RKD-D, RKD-A, RKD-DA 로 구분하고, 다른 손실함수와 결합해서 사용할 경우 항상 각 손실함수의 조정계수(balancing factor)를 고려했다고 합니다. 각 태스크에 대하여 RKD를 FitNet, Attention, HKD (Hinton's KD), Dark-Rank 등과 비교했고, 하이퍼파라미터의 공정한 비교를 위해 grid search로 최적화했습니다.

<span style="font-size:7pt"> *Dark-Rank = 데이터 사이의 유사도 순위를 transfer하는, metric learning에 적합한 KD 방법 (metric learning task에서만 사용) </span>



<br/>

### 4.1 Metric learning

여러 다른 task들 중에서 relational knowledge와 가장 관련있는 태스크라고 볼 수 있음

metric learning은 data examples들을 하나의 매니폴드로 projects하는 embedding model을 train하는데 목표가 있다.

- 이 manifold 두 exmaples가 의미상으로(sementically) 비슷한 경우, 서로 가까운 metric을 갖고, 다른 경우 멀리 떨어지게 함

- Metric learning aims to train an embedding model that projects data ex- amples onto a manifold where two examples are close to each other if they are semantically similar and otherwise far apart. 

- embedding 모델들은 일반적으로 image retrieval (이미지 검색)에 대해 평가하므로, 다음과 같이 data set을 고르고 validate함
  - CUB-200-2011 [40], Cars 196 [14], Stanford Online Products [21] datasets



- evaluation metric = recall@K
  - 모든 test images가 모델을 통해 embedding되고 나면, 각각의 테스트 이미지를 쿼리로서 사용해, 쿼리를 제외한 테스트 셋에서 top K-nearest neighbor images를 검색함
  - 검색된 이미지들이 쿼리와 똑같은 카테고리일 경우, query에 대한 recall은 1이라고 봄
  - Recall@K는 모든 테스트 셋에 대해 average recall을 가져옴

- 학습할 때는 [42]의 프로토콜을 따름
  - resized된 256$$\times$$256 이미지들로부터 랜덤 cropping 해서
    224$$\times$$224 크기의 training samples를 얻은 후,
    data augementation을 적용함 (random horizontal flipping)
  - 평가할 땐, single center crop을 사용
  - 모든 데이터셋들은 128 batch size로 Adam optimizer를 사용해 training
  - effective pairing을 위해, [31] FaceNet에서 나온 batch construction을 따라했고, 미니 배치에서 카테고리마다 5가지의 postivie images를 샘플링했음
- Teacher model은 ResNet50 사용, ImageNet ILSVRC dataset을 pre-trained한 모델
  - 신경망의 avgpool까지의 레이어들을 가져왔고, single FC 레이어를 뒤에 붙임(512짜리 embedding size) 그리고 난 뒤, l2-normalization 해 줌
- Student model은 ResNet18 사용, ImageNet-pretrained
  - teacher과 비슷한 방식으로 구성, embedding 사이즈만 다르게
- Teacher model은 triplet loss로 학습시킴 (most common and effective in metric learning)



- **Teacher model = triple loss [31]**
  - anchor, positive, negative
  - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127005133322.png" alt="image-20200127005133322" style="zoom:50%;" />
  - triplets을 뽑을 때, distance-weighted sampling [42]
  - embedding vector들이 unit length를 가질 수 있게끔, final embedding layer에 $$l2$$ normalization 적용
    - l2-normalization을 사용하면, [0, 2]로 embedding points 사이 거리의 범위를 제한시킬 수 있어서, triplet loss의 학습을 안정시킬 수 있다고 알려짐
    - l2- normalization을 embedding에 사용하는 것은 deep metric learning [7, 13, 21, 22, 34, 41] 분야에서 넓게 사용됨
- **RKD **
  - teacher와 student의 final embedding outputs에다가 RKD-D, RKD-A를 적용함
  - 제안된 RKD로스들은 embedding points 사이의 거리의 범위에 영향을 딱히 받지 않기 때문에 marin이나 triplet sampling parameters와 같은 민감한 파라미터들이 필요없다는 장점이 있음
  - RKD의 robustness를 보여주기 위해, RKD에 l2 normalization의 적용 유무를 비교함
  - RKD loss들을 사용한 metric learning에서, triplet loss와 같은 task loss를 사용하지 않음
    - 왜냐하면, 모델이 원래의 ground-truth labels없이 순수 teacher model의 가이드에 의해 학습되어야 하므로... (실험에서, task loss를 추가적으로 사용하는 것은 의미가 없었음)
- Attention, FitNet, DarkRank

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127164642541.png" alt="image-20200127164642541" style="zoom:50%;" />

<br/>

#### 4.1.1 Distillation to smaller networks

- student Baseline = directyl trained with Triplet

- RKD는 embedding dimension에 영향을 덜 받음 , Triplet은 영향이 큼
  - relative gain of recall@1 값 비교해보면, 

- RKD는 embedding space를 잘 활용하기 때문에, l2-normalization 없이 훈련시키는 것이 더 잘 나온 것을 확인할 수 있음, 다른 방법들에서는 l2-normalization이 없는 경우 성능이 더 낮아짐

- **Cars 196 데이터셋에서는 smaller backbone과 less embedding dimension을 가진 students가 RKD에 의해 teacher model을 뛰어넘는 성능을 보이고 있음**

<br/>

#### 4.1.2 Self-distillation

- RKD가 청출어람(students모델이 teacher보다 transfer을 통해 뛰어나게 만듬)라는 점을 관찰했기 때문에, self-distillation 실험을 진행했음 (students 모델 아키텍쳐를 teacher과 동일하게 만듬), 실험결과로 보았을 때 당연히 l2-normalization 적용안함

- RKD-DA로 전 세대를 학습(이전 세대의 student를 새로운 Teacher로 학습) = The model at Gen($$n$$) is guided by the model at Gen($$n-1$$).

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127170223095.png" alt="image-20200127170223095" style="zoom:50%;" />


표에서 보면, 자가 증류(Self-distillation)를 통해 학습한 여러 세대의 Student 모델들은 초기 세대의 Teacher 모델(Triplet)보다 성능이 더 좋습니다. 그러나 2번째 세대(@Gen2)부터는 성능이 향상되지 않습니다.

<br/>

#### 4.1.3 Comparison with state-of-the art models

metric learning의 최근 SOTA 방법들은 GoogLeNet을 백본으로써 차용함, [42]만 ResNet50의 변형을 사용(채널 수만 수정)

공정한 비교를 위해, student models을 GoogLeNet과 ResNet50로 모두 학습시켰고, 다른 방법들과 같은 임베딩 사이즈를 사용

RKD-DA는 student models를 trainig하는데 사용함

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127170817085.png" alt="image-20200127170817085" style="zoom:50%;" />

<br/>

#### 4.1.4 Discussion

**RKD는 l2-normalization이 없을 때 성능이 더 좋음**

- RKD over Triplet(?) 는 l2 normalization 없이도 안정적으로 학습될 수 있다는 장점이 있음
- l2 norm은 임베딩한 모델의 output points를 unit-hypershpere의 표면으로 강제적으로 놓게 함
- 따라서 l2-norm 없는 student 모델은 임베딩 공간을 fully utilize할 수 있게 됨 -> 이점이 RKD가 table 1에서 좋은 이유
- Dark-Rank는 triplet loss를 포함하는데, 이 triplet loss는 l2-norm 없이는 굉장히 취약한 것은 알려진 사실임
  - 77.00 with l2-norm 인데 , 없으면 52.92까지 떨어짐

**Students excelling teachers.**

- [2, 9, 45]와 같은 분류에서도 비슷한 효과가 보여진 바가 있음
  - [2, 9] 에서는 teacher의 class distribution의 soft output이 추가적인 정보를 전달한다고 설명하고 있다. (예를 들면, cross-category relationships, 이는 ground-truth labels를 원핫백터로 해석될 수 없는)
- 마찬가지로, RKD의 continuous target labels (ex, distance or angle)는 유용한 정보를 전달하는데, 이는 triplet loss와 같은 conventional loss에서 사용된 binary(positive/negative) ground-truth labels로 해석될 수 없는 정보들.

**RKD as a training domain adaptation**

- Cars 196과 CUB-200-2011 데이터셋은 원래 fine-grained classfication을 위해 디자인되었음
  - 이는 severe intra-class variations와 inter-class similiarity 때문에 굉장히 challenging함
- 이러한 데이터셋들에 대해서는, 도메인의 성격을 구체화하기 위해 효과적인 adaptation은 중요함
  - fine-grained classification에 대한 최근 메서드들은, target-domain objects의 localizing discriminative parts에 초점을 두고 있는 연구들이다. [23, 44, 48]
- RKD 로스들로 훈련한 하나의 모델의 adaptation 정도를 측정하기 위해, 하나의 학습 데이터 도메인과, 다른 데이터 도메인들에 놓여진 것들과 recall@1값을 비교함
- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127173009810.png" alt="image-20200127173009810" style="zoom:30%;" />
- Cars 데이터셋에 대해 훈련된 student model을 이용해 다른 datasets들에 대해 recall@1값을 비교함
- teacher(Triplet)의 recall@1값은 초기 모델의 pretrained feature와 비슷하게 유지된 반면에, student(RKD)는 different domains에 대해 낮은 recall@1 값을 보임
- RKD는 다른 도메인들에 대해 일반화를 가지는 능력을 희생해, 학습 도메인에만 모델을 강력하게 adapts하는 것을 알 수 있음

<br/>

### 4.2 Image classification

> 
>
> Image Classification 결과 표
>
> 

사용한 데이터셋은 CIFAR-100과 Tiny ImageNet이고, RKD와 비교 대상으로 IKD, HKD, FitNet, Attention을 사용했습니다. 모두 cross-entropy loss가 포함되어 있고, ResNet과 VGG를 기반으로 하여 Teacher/Student 모델을 구성했습니다.

표에서 볼 수 있듯이, RKD-DA와 HKD를 함께 사용한 방법이 가장 성능이 좋습니다. 또한, RKD-DA와 결합했을 때 대부분의 경우 성능이 향상됩니다.

<br/>

### 4.3 Few-shot learning

- Few-shot learning의 목표 = to learn a classifier that generalizes to new unseen classes with only a few examples for each new class
- 실험세팅

RKD의 비교대상으로 few-shot classification에서 standard benchmarks인 Omniglot, miniImageNet을 사용했습니다.



- prototypical networks [33]를 이용해 RKD를 평가함
  - prototypical networks = 분류가 새로운 클래스들의 주어진 examples들의 거리기반으로 수행되는, 임베딩 네트워크들을 학습함
  - prototypical networks를 4개의 convolutional layers로만 구성된 shallow networks를 구성했기 때문에, 더 작은 network를 student로 사용하기 보다, self-distillation을 사용함 (교사와 제자의 아키텍쳐 동일)
- teacher/ student network의 final embedding output에 RKD, FitNet, Attention 적용
- RKD-D, RKD-A를 결합해 사용한 경우, 최종 loss를 2로 나눔
- 모든 세팅에 대해, final loss에 prototypical loss를 더함

- few-shot classification에서 흔하게 evaluation protocol에 따름 [33]

  - Omniglot - 1000개의 랜덤 생성한 에피소드들에 대해 평균치를 내서 정확도 계산
  - MiniImagenet - 600개 랜덤 생성한 에피소드들에 대해 평균치로 정확도 계산
- 결과
- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127180256095.png" alt="image-20200127180256095" style="zoom:50%;" />
- The Omniglot results are summarized in Ta- ble 5 while the *mini*ImageNet results are reported with 95% confidence intervals in Table 6.
- 결국 우리 방법이 teacher를 뛰어 넘는 student 성능을 지속적으로 보여줌

<br/>

<br/>

## 5. Conclusion

We have demonstrated on different tasks and bench- marks that the proposed RKD effectively transfers knowl- edge using mutual relations of data examples. In particular for metric learning, RKD enables smaller students to even outperform their larger teachers. While the distance-wise and angle-wise distillation losses used in this work turn out to be simple yet effective, the RKD framework allows us to explore a variety of task-specific RKD losses with high- order potentials beyond the two instances. We believe that the RKD framework opens a door to a promising area of effective knowledge transfer with high-order relations.
<br/>

<br/>

---



## 99. 나의 생각















































