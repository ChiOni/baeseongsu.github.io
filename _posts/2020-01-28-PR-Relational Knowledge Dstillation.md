---
layout: post
title:  "Paper Review : Relational Knowledge Distillation"
date:   2020-01-28 22:22
categories: [Deep Learning, Knowledge Distillation]
use_math: true
---

안녕하세요. 배성수입니다.

오늘 리뷰할 논문은 CVPR 2019에 Accepted Paper인 ***Relational Knowledge Distillation*** 입니다.

최근 딥러닝 스터디를 하며 Knowledge Distillation(KD) 관련 논문 2편을 읽었습니다. NIPS 2014 Deep Learning Workshop에서 발표한 ***Distilling the Knowledge in a Neural Network*** , ICML 2019에 Accepted Paper인 ***Zero-Shot Knowledge Distillation in Deep Networks*** 입니다. 극히 주관적인 저의 생각입니다만, 앞선 2편의 KD 논문에 비해 오늘 리뷰할 논문이 좀 더 명확하고 깔끔했습니다. 특히 핵심 아이디어가 뚜렷하고, 뒷받침하는 실험들의 세팅이 탄탄하다는 느낌을 받았습니다.

저자는 POSTECH의 Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho 입니다. 재밌게 잘 읽었습니다. 감사합니다.



---

## Abstract

- Conventional KD Approach
  - as a from of training the student to mimic output activations of **individual data examples** represented by teacher
- Our Approach (Relational Knowledge Distillation)
  - transfers **mutual relations of data examples**
- distance-wise & angle-wise distllation losses를 제안
  - penalize structural differences in relations
- achieve SOTA

## 1. Introduction

최근 Computer Vision

- 최근 CV나 AI쪽 연구에서는 많은 연산량과 메모리를 필요로 하는 모델들이 등장
- 이 물리적 부담을 줄이기 위한 방법 중 하나로 knowledge를 전달하는 방식이 있음
- 이에 대해, 결국 중요한 두 가지 질문이 존재
  - (1) What constitutues the knowledge in a learned model?
  - (2) How to transfer the knowledge into another model?
- [3,4,11]과 같은 transfer methods의 assumption
  - knowledge = learned mapping from inputs to outputs
  - transfer = teacher's outputs을 student model의 training targets로 학습시킴
- 최근 연구는 이렇습니다 ~
  - [1,11,12,27,47] very effective for training a student model
  - [2, 9, 45] improve a teacher model itself by self-distillation
- KD를 linguistic structuralism [19] 관점에서 본다면
  - = semiological system 내에서 structural relations에 초점을 맞춰 본다면
  - Saussure’s concept of the relational identity of signs is at the heart of structuralist the- ory; “In a language, as in every other semiological system, what distinguishes a sign is what constitutes it” [30]. In this perspective, the meaning of a sign depends on its relations with other signs within the system; a sign has no absolute meaning independent of the context.
- 이분들의 work의 central tenet는 knowledge라는 것이 개별적인 학습된 representations보다 학습된 represenations의 관계에 의해 더 잘 표현되어진다.라는 것임
  - individual data exmaple : an image
  - 다른 data examples와 비슷하거나 대조적인 representation이라는 의미를 얻을 수 있음
  - 그러한 주요 정보들은 data embedding space안에서 structure로 놓여질 수 있음
  - 그렇기 때문에, KD에 대해 RKD라는 novel approach를 제안함
  - 이는, 개별 outputs보다 각 outputs의 structural relations를 transfer하는 방식으로 이루어짐
  - 좀 더 구체적인 realizations를 위해, 두가지 RKD 로스를 제안
    - Distance-wise (second-order) distillation loss
    - Angle-wise (third-order) distillation loss
  - RKD는 일반적인 KD의 generalization으로 볼 수 있고, 성능을 끌어올리기 위해 다른 방법들과 결합할 수 도 있음
    - due to its complementarity with conventional KD ???
  - metric learning, image classification, few-shot learning 에서 이 연구가 student models의 성능을 상당히 향상시킴
  - 결국, 이 실험들을 통해 knowledge가 relation 속에 살고 있고, RKD는 knowledge를 전파하는데 있어 효과적인 방법임



## 2. Related Work

한 모델의 지식(Knowledge)을 다른 모델로 전달(Transfer)하는 연구는 꽤 오랫동안 연구되었다고 합니다. 처음으로, Breiman and Shang이 트리 기반의 model compression을 통해 지식을 전달하는 방법을 제안했다고 합니다. 그 이후로, 신경망으로 model compression이 넘어왔고, Hinton 교수님이 soft targets라는 컨셉을 이용하여 지식 증류(Knowledge Distillation)라는 네이밍을 탄생시켰습니다. 최근에는 HKD(Hinton's KD)를 이은 후속 연구들뿐만 아니라 기존 접근과 다른 방식의 연구가 진행되고 있으며, 지도학습을 넘어 준지도학습/비지도학습 영역에서의 KD, 테스크에 특화된 KD 등에 대한 연구가 진행되고 있습니다.

다양한 KD 연구흐름 속에서 Chen의 연구가 rank loss를 사용해 similarities를 transfer하는 metric learning 기반의 KD라는 점에서 본인들의 연구와 어느정도 유사성이 있다고 말합니다. 그러나, Chen의 연구는 metric learning에만 제한되어 있고, 본 연구는 다양한 테스크에 적용가능한 general framework라고 말합니다. 게다가, metric learning task에서 Chen 것보다 성능이 더 좋았다고 합니다.

## 3. Our Approach

먼저, conventional KD 보고, RKD의 general form에 대해 소개함

간단하지만 효과적인 두 가지 distillation losses를 소개함

Notation

- $$T, S$$
- $$f_{T}, f_{S}$$
- $$\chi^{N}$$ : a set of $$N$$-tuples of distinct data examples
  - ex) $$\chi^{2} = \{ (x_i, x_j) | i \neq j \} $$
    $$\chi^{3} = \{ (x_i, x_j, x_k) | i \neq j \neq k \} $$



### 3.1 Conventional KD

- [1, 2, 8, 1,, 12, 25, 27, 45, 47] = conventional KD

  - minmizing the objective function like below

    <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200126174402023.png" alt="image-20200126174402023" style="zoom:50%;" />

  - $$l$$  = loss function

  - penalizes the difference between the teacher and the student

  - [11] Hinton et al. 은 pre-softmax outputs에 대해 temperature $$\tau$$를 적용한 뒤 softmax를 먹인 후 Kullback-Leibler divergence for $$l$$

    - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200126174627447.png" alt="image-20200126174627447" style="zoom:50%;" />

  - [27] Romero et al. 은 은닉층의 아웃풋들을 $$f_T$$ 와 $$f_S$$로 설정하고 hidden activations의 지식을 propagates함, 그리고 loss function은 squared Euclidean distance임. 일반적으로 $$S$$와 $$T$$의 은닉층 출력의 차원이 다르기 때문에, student network에 $$\beta$$ 매핑을 통해 서로 다른 차원을 bridge해주는 역할을 수행함

    - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200126175131540.png" alt="image-20200126175131540" style="zoom:50%;" />

- [1, 2, 8, 12, 25, 45, 47]과 같은 많은 방법들은 앞에서 본 (1)식과 같이, teacher network와 student network의 mapping 값의 loss를 줄여나가는 방식으로 볼 수 있음

- 본질적으로, conventional KD는 individual한 teacher의 outputs을 student에게 전달해줌

- 이러한 종류의 KD methods를 Individual KD (IKD)라고 명명함



### 3.2 Relational KD



RKD는 teacher's output presentation에서 data examples의 mutual relations를 이용해 구조적 knowledge를 전달하는 것을 목표로 함

기존 KD와 다르게, RKD는 각각의 n-튜플의 데이터들에 대한 relational potential $$\psi$$ 를 계산하고 그 포텐셜 값을 통해 정보를 teacher에서 student로 전달함

- Define $$t_i = f_{T}(x_{i})$$ , $$s_{i} = f_{S}(x_{i})$$
- RKD objective는 다음과 같이 표현됨
  - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200126234145205.png" alt="image-20200126234145205" style="zoom:50%;" />
  - $$\psi$$ - relational potential function
    - measures a relational energy of the given n-tuple

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



#### 3.2.1 Distance-wise distillation loss

\psi_{\text{D}} 라는 거리 기반의 포텐셜 함수(distance-wise potential function)를 \psi_{\text{D}} (t_{i},t_{j}) = \frac{1}{\mu}{||t_{i}-t_{j}||\mathunderscore{2}} 라고 정의합니다. 즉, 한 쌍을 이루는 두 개의 데이터 샘플이 신경망을 통해 output representation space에 놓여질 때, 그들간의 유클리디안 거리를 계산하는 함수라고 보시면 됩니다. 여기서 $\mu$ 는 거리함수의 normalization factor 입니다. 그렇다면, 이 $\mu$ 는 어떻게 정하는 것이 좋을까요?

논문의 핵심 아이디어가 결국 관계성에 있기 때문에, 다른 쌍들과 비교하여 상대적 거리를 계산하는데 초점을 맞추게 됩니다. 따라서 쌍으로 구성된 미니배치인 $\chi^{2}$ 에서 나온 각각의 페어 데이터의 평균 거리로 계산하게 됩니다. 이를 수식으로 나타내면, $\mu = \frac{1}{|\chi^{2}|}{\sum_{(x_{i}, x_{j})\in\chi^{2}}{||t_{i}-t_{j}||_{2}}}$ 라고 표현할 수 있습니다.

만약 $\mu$ 와 같은 factor가 존재하지 않는다면, Teacher 모델의 dimension이 일반적으로 더 크기 때문에 Teacher 모델과 Student 모델 사이의 거리 scale 차이가 발생하게 됩니다. 따라서 논문에서는 $\mu$ 를 사용하여 $\psi_{\text{D}}$ 라는 포텐셜 함수가 결국 distance-wise potentials를 잘 반영할 수 있도록 합니다. 실제로 $\mu$ 라는 factor로 인해 학습이 더 안정적이고 빠르게 수렴하는 것을 관찰했다고 합니다.

위를 통해 



distance-wise distillation loss를 다음과 같이 정의

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000753806.png" alt="image-20200127000753806" style="zoom:50%;" />

- $$l_{\delta}$$는 Huber loss를 의미함 (외부에선 MAE, 이상치 덜 민감 - 내부에선 세밀하게 MSE)
  - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000832781.png" alt="image-20200127000832781" style="zoom:50%;" />
- 결국, output representation spaces 사이의 거리차를 penalizing함으로써 데이터 쌍의 관계를 transfer함
- 기존 KD와 다르게, teacher output을 직접적으로 student와 매치하게 강요하는 것이 아니라, outputs의 거리 구조에 초점을 두도록 학습시킨다.



#### 3.2.2 Angle-wise distillation loss

앞에서 $\psi_{\text{D}}$ 를 잘 이해했다면 angle-wise relational potential function 

- 세 쌍이 주어진 경우, output representation space에서 세 가지 값이 만든 angle을 measure함
- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127001618420.png" alt="image-20200127001618420" style="zoom:50%;" />

angle-wise distillation loss

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127001654372.png" alt="image-20200127001654372" style="zoom:50%;" />

- angular differences를 penalizing해서 training example embeddings의 관계를 transfer함
- 거리보단 각이 더 higher-order property 이기 때문에, 학습 과정에서 relational information을 student에게 더욱 효과적이고 유연하게 transfer할 수 있음
- 실험에서, angle-wise loss가 종종 더욱 빠르게 수렴하고, 더 좋은 성능을 보이는 것을 관찰함



#### 3.2.3 Training with RKD

학습과정에서 제안된 RKD 손실함수는 단독으로 사용할 수도 있고,  task에 특화된 손실함수와 함께 사용할 수도 있습니다. 따라서 전체적인 목적함수(objective)를 수식으로 표현하면 다음과 같은 형태가 됩니다.

$$\mathcal{L}_{\text{task}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}}$$



- $$\lambda_{KD}$$는 loss 항들의 밸런스를 맞추기 위한 tunable hyperparameter임
- multiple KD losses로 학습될 때, 각 로스는 해당하는 balancing factor로 가중치가 부여됨
- RKD에서 제안된 distillation loss들을 구할 때, tuple을 뽑는 샘플링은 주어진 미니배치 속 examples에 대해 가능한 모든 조합에 대해 simply 사용



#### 3.2.4 Distillation target layer

RKD에서 distillation target function $$f$$ 는 이론상으로 아무 레이어의 output mapping으로 생각할 수 있다. 하지만, distance/angle-wise losses는 teacher의 개별적인 outputs를 transfer하지 않기 때문에, 개별 output 값들 자체가 중요한 곳에 혼자 사용하는 것은 적절치 않다. Ex) softmax layer for classification

그런 경우에는, IKD 로스나 task-sepcific 로스를 함께 사용하는 것이 필요

그 밖에 대부분의 다른 경우는 RKD가 적용가능하고 효과적이었음



## 4. Experiments

metric learning, classification, few-shot learning 이라는 3가지 태스크에 대해 실험을 진행했습니다. 기존의 RKD를 사용한 손실함수에 따라 RKD-D, RKD-A, RKD-DA 로 구분하고, 다른 손실함수와 결합해서 사용할 경우 항상 각 손실함수의 조정계수(balancing factor)를 고려했다고 합니다.

각 태스크에 대하여 RKD를 FitNet, Attention, HKD (Hinton's KD), Dark-Rank 등과 비교했고, 하이퍼파라미터의 공정한 비교를 위해 grid search로 최적화했다고 합니다. 

*Dark-Rank = 데이터 사이의 유사도 순위를 transfer하는, metric learning에 적합한 KD 방법 (metric learning task에서만 사용)



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
  - margin (m) = 0.2
  - triplets을 뽑을 때, distance-weighted sampling [42]
  - embedding vector들이 unit length를 가질 수 있게끔, final embedding layer에 $$l2$$ normalization 적용
    - l2-normalization을 사용하면, [0, 2]로 embedding points 사이 거리의 범위를 제한시킬 수 있어서, triplet loss의 학습을 안정시킬 수 있다고 알려짐
    - l2- normalization을 embedding에 사용하는 것은 deep metric learning [7, 13, 21, 22, 34, 41] 분야에서 넓게 사용됨
- **RKD **
  - teacher와 student의 final embedding outputs에다가 RKD-D, RKD-A를 적용함
  - 제안된 RKD로스들은 embedding points 사이의 거리의 범위에 영향을 딱히 받지 않기 때문에 marin이나 triplet sampling parameters와 같은 민감한 파라미터들이 필요없다는 장점이 있음
  - RKD의 robustness를 보여주기 위해, RKD에 l2 normalization의 적용 유무를 비교함
  - RKD-DA에 대해서는 lambda_RKD-D = 1, lambda_RKD-A =2 로 세팅
  - RKD loss들을 사용한 metric learning에서, triplet loss와 같은 task loss를 사용하지 않음
    - 왜냐하면, 모델이 원래의 ground-truth labels없이 순수 teacher model의 가이드에 의해 학습되어야 하므로... (실험에서, task loss를 추가적으로 사용하는 것은 의미가 없었음)
- Attention, FitNet, DarkRank

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127164642541.png" alt="image-20200127164642541" style="zoom:50%;" />

#### 4.1.1 Distillation to smaller networks

- student Baseline = directyl trained with Triplet

- RKD는 embedding dimension에 영향을 덜 받음 , Triplet은 영향이 큼
  - relative gain of recall@1 값 비교해보면, 

- RKD는 embedding space를 잘 활용하기 때문에, l2-normalization 없이 훈련시키는 것이 더 잘 나온 것을 확인할 수 있음, 다른 방법들에서는 l2-normalization이 없는 경우 성능이 더 낮아짐

- **Cars 196 데이터셋에서는 smaller backbone과 less embedding dimension을 가진 students가 RKD에 의해 teacher model을 뛰어넘는 성능을 보이고 있음**



#### 4.1.2 Self-distillation

- RKD가 청출어람(students모델이 teacher보다 transfer을 통해 뛰어나게 만듬)라는 점을 관찰했기 때문에, self-distillation 실험을 진행했음 (students 모델 아키텍쳐를 teacher과 동일하게 만듬), 실험결과로 보았을 때 당연히 l2-normalization 적용안함

- RKD-DA로 전 세대를 학습(이전 세대의 student를 새로운 Teacher로 학습) = The model at Gen($$n$$) is guided by the model at Gen($$n-1$$).

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127170223095.png" alt="image-20200127170223095" style="zoom:50%;" />

- 결과를 보면, 첫 세대의 teacher(triplet)보다도 성능이 더 좋아진 것을 확인 가능함

- 특히, CUB와 Cars는 꽤 큰 차이로 성능이 좋아짐

- 그러나 2번째 세대부터 성능이 향상되진 않음

  

#### 4.1.3 Comparison with state-of-the art models

metric learning의 최근 SOTA 방법들은 GoogLeNet을 백본으로써 차용함, [42]만 ResNet50의 변형을 사용(채널 수만 수정)

공정한 비교를 위해, student models을 GoogLeNet과 ResNet50로 모두 학습시켰고, 다른 방법들과 같은 임베딩 사이즈를 사용

RKD-DA는 student models를 trainig하는데 사용함

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127170817085.png" alt="image-20200127170817085" style="zoom:50%;" />

- 우리께 항상 좋지 않고, ABE8 [13] 이 CARS와 SOP 데이터셋에 대해 가끔 좋은 이유
  - 우리건 구글넷에서 single embedding layer를 사용하지만,
  - ABE8 [13]은 각 branches마다 추가적인 multiple attention modeuls를 필요로 하기 때문...

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



### 4.2 Image classification

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127174449521.png" alt="image-20200127174449521" style="zoom:50%;" />

사용한 데이터셋은 CIFAR-100과 Tiny ImageNet이고, RKD와 비교 대상으로 IKD, HKD, FitNet, Attention을 사용했습니다. 모두 cross-entropy loss가 포함되어 있고, ResNet과 VGG를 기반으로 하여 Teacher/Student 모델을 구성했습니다.

RKD-DA와 HKD를 함께 사용한 방법이 가장 성능이 좋았습니다. 대부분의 경우 RKD-DA를 붙였을 때 성능이 향상되는 것을 보면, RKD는 다른 방법을 보완하는 역할을 수행한다고도 볼 수 있겠습니다.



### 4.3 Few-shot learning

- Few-shot learning의 목표 = to learn a classifier that generalizes to new unseen classes with only a few examples for each new class

- 실험세팅
  - 실험에 standar benchmarks for few-shot classification 이용
    - Omniglot [16], miniImageNet [39]
  - prototypical networks [33]를 이용해 RKD를 평가함
    - prototypical networks = 분류가 새로운 클래스들의 주어진 examples들의 거리기반으로 수행되는, 임베딩 네트워크들을 학습함
    - data-aug, training procedure을 Snell et al. [33] 에 따름
    - [39] Vinyals 가 제시한 split 방법을 사용
    - prototypical networks를 4개의 convolutional layers로만 구성된 shallow networks를 구성했기 때문에, 더 작은 network를 student로 사용하기 보다, self-distillation을 사용함 (교사와 제자의 아키텍쳐 동일)
  - teacher/ student network의 final embedding output에 RKD, FitNet, Attention 적용
  - RKD-D, RKD-A를 결합해 사용한 경우, 최종 loss를 2로 나눔
  - lambda_Attention = 10
  - 모든 세팅에 대해, final loss에 prototypical loss를 더함

- few-shot classification에서 흔하게 evaluation protocol에 따름 [33]

  - Omniglot - 1000개의 랜덤 생성한 에피소드들에 대해 평균치를 내서 정확도 계산
  - MiniImagenet - 600개 랜덤 생성한 에피소드들에 대해 평균치로 정확도 계산

- 결과

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127180256095.png" alt="image-20200127180256095" style="zoom:50%;" />

- The Omniglot results are summarized in Ta- ble 5 while the *mini*ImageNet results are reported with 95% confidence intervals in Table 6.

- 결국 우리 방법이 teacher를 뛰어 넘는 student 성능을 지속적으로 보여줌

  

## 5. Conclusion

We have demonstrated on different tasks and bench- marks that the proposed RKD effectively transfers knowl- edge using mutual relations of data examples. In particular for metric learning, RKD enables smaller students to even outperform their larger teachers. While the distance-wise and angle-wise distillation losses used in this work turn out to be simple yet effective, the RKD framework allows us to explore a variety of task-specific RKD losses with high- order potentials beyond the two instances. We believe that the RKD framework opens a door to a promising area of effective knowledge transfer with high-order relations.



---



## 99. 나의 생각









