# Relational Knowledge Distillation

CVPR 2019

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

knowledge를 한 모델에서 다른 모델로 전달하는 연구는 꽤 오랫동안 연구되어 왔음

[3] Breiman and Shang이 처음으로 multiple-tree model들의 성능을 approximate하면서 더욱 해석력있게 만들도록 single-tree models를 학습시키는 방법을 제안함

[4] Bucilua et al. [1] Ba and Caruana [11] Hinton et al. - NN에서 비슷한 접근을 하기 시작했음, 주로 model compression의 목적으로 연구함

[4] Bucilua et al. 은 NN의 신경망 모델들의 앙상블을 단일 신경망으로 compress함

[1] Ba and Caruana는 a shallow NN의 정확도를 향상시킴, Deep NN을 따라하도록 훈련시킴으로써, (두 네트워크 사이의 로짓값들의 차이를 penalizing하는 방식으로)

[11] Hinton et al. KD라는 이름을 탄생시키며, student model을 teacher model의 softmax 분포와 matching하는 objective로서

최근에, 이들을 이은 subsequent 후속 paper들이 KD와 다른 접근들을 제안하며 등장함

- [27] Romero et al. 는 relatively narrower students를 학습하기 위해 addtional linear projection layers를 사용함으로써 a teacher model을 distill함
- [47] Zagoruyko and Komodakis, [12] Huang and Wang은 teacher network의 attention map을 student에게 transfer하는 방식, [36] Tarvainen and Valpola 는 mean weights를 이용해서 비슷한 방식으로 접근
- [17] Lopes et al. 은 teacher model의 메타데이터를 이용하는 data-free KD를 제안함, 반면에 [29] Sau et al. 은 KD에 noise-based regularizer를 제안함
- [43] Xu et al. 은 KD의 loss function을 학습하기 위해 conditional adversarial network를 제안함
- [8] Crowley et al. 는 모델의 convolutional channels들을 그룹핑해서 attention transfer과 함께 모델을 학습시킴으로써 model을 compress함
- [25] Polino et al. [20] Mishra and Marr 는 KD를 network quantization을 결합했는데, 이는 weights와 activations의 bit precision을 줄이는데 도움을 줌

최근 연구

- [2, 9, 45] 는 teacher model을 동일한 아키텍쳐의 student model로 distilling 함으로써 teacher 보다 student를 향상시키는 것을 보임 (= self-distillation)
- 특히, [9] Furlanello et al. & [2] Bagherinezhad et al. 는 teacher의 softmax outputs을ground truch over generations로서 사용함으로써 studen를 학습시킴으로써 위를 입증함
- [45] Yim et al. 은 Gramian matrices를 사용해 output activations를 transfer하고 난뒤 student모델을 fine-tune하는 방식임
- We also demonstrate that RKD strongly benefits from self-distillation.?

supervised learning을 넘어서서 KD를 연구하고 있음

- [11, 38]의 두가지 프레임워크를 통합시켜 [18] Lopez-Paz et al. 은 unsupervised, semi-supervised, and multi-task learning scenarios로 확장함
- [26] Radosavovic et al. 은 multiple data transformations를 적용해 하나의 example로부터 multiple predictions를 generate하고 난 뒤, omni-supervised learning을 위해 annotations로서 predictions의 앙상블로서 사용

KD에 대한 연구가 진행됨에 따라, task-specific KD method가 등장함

- [5, 6, 37] object detection
- [24] face model compression
- [7] image retrieval and Re-ID
- 특히, [7] Chen et al. 의 연구는 rank loss를 사용해 images들 간의 similarities를 transfer하는 metric learning 방식의 KD technique을 제안함 - ranks의 relational information을 전달한다는 점에서, 우리 연구와 어느정도 유사성이 있음
- 그러나, Chen의 연구는 metric learning에만 제한되어 있고, 우리는 RKD라는 일반적인 framework를 제안하며, 다양한 태스크들에 대해 적용가능성을 입증함.
- 게다가, metric learning에서 우리 실험은 Chen것보다 성능이 outperform with a significant margin 임



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

$$\psi_{D}$$ = distance-wise potential function

- Output representation space에서 두 개의 샘플들의 Euclidean distance를 의미함
- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000148102.png" alt="image-20200127000148102" style="zoom:50%;" />
- $$\mu$$는 distance의 normalization factor
  - 다른 쌍들 사이에서 relative distances에 대해 초점을 맞추고 싶기 때문에, $$\mu$$를 미니배치 안에서 돌아가는 $$\chi^2$$ = 페어셋으로부터 나온 페어들간의 평균 거리로 설정함
  - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000356493.png" alt="image-20200127000356493" style="zoom:50%;" />
  - teacher model의 uclidean distance와 student model의 uclidean distance 사이에 크기 차이가 있을을 수 있기 때문에$$\mu$$값을 저렇게 평균값으로 설정함으로써 teacher student 사이의 distance-wise potentials를 matching할 수 있게 댐
    - scaling 차이가 일어나는 이유로 dimension이 다른 경우...
  - 실험을 하면서, normalization factor로 학습이 더 안정적이고 빠르게 수렴하는 것을 관찰함

distance-wise distillation loss를 다음과 같이 정의

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000753806.png" alt="image-20200127000753806" style="zoom:50%;" />

- $$l_{\delta}$$는 Huber loss를 의미함 (외부에선 MAE, 이상치 덜 민감 - 내부에선 세밀하게 MSE)
  - <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127000832781.png" alt="image-20200127000832781" style="zoom:50%;" />
- 결국, output representation spaces 사이의 거리차를 penalizing함으로써 데이터 쌍의 관계를 transfer함
- 기존 KD와 다르게, teacher output을 직접적으로 student와 매치하게 강요하는 것이 아니라, outputs의 거리 구조에 초점을 두도록 학습시킨다.



#### 3.2.2 Angle-wise distillation loss

angle-wise relational potential

- 세 쌍이 주어진 경우, output representation space에서 세 가지 값이 만든 angle을 measure함
- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127001618420.png" alt="image-20200127001618420" style="zoom:50%;" />

angle-wise distillation loss

- <img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127001654372.png" alt="image-20200127001654372" style="zoom:50%;" />

- angular differences를 penalizing해서 training example embeddings의 관계를 transfer함
- 거리보단 각이 더 higher-order property 이기 때문에, 학습 과정에서 relational information을 student에게 더욱 효과적이고 유연하게 transfer할 수 있음
- 실험에서, angle-wise loss가 종종 더욱 빠르게 수렴하고, 더 좋은 성능을 보이는 것을 관찰함



#### 3.2.3 Training with RKD

학습할 때, 제안한 RKD loss들을 포함해서, multiple distillation loss functions는 단독으로 사용할 수도 있고, task-specific loss functions와 함께 사용할 수도 있음 (예를 들면, CE for classification)

따라서, overall objective는 다음과 같은 최종 형태가 됨

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127002325341.png" alt="image-20200127002325341" style="zoom:50%;" />

- $$\lambda_{KD}$$는 loss 항들의 밸런스를 맞추기 위한 tunable hyperparameter임
- multiple KD losses로 학습될 때, 각 로스는 해당하는 balancing factor로 가중치가 부여됨
- RKD에서 제안된 distillation loss들을 구할 때, tuple을 뽑는 샘플링은 주어진 미니배치 속 examples에 대해 가능한 모든 조합에 대해 simply 사용



#### 3.2.4 Distillation target layer

RKD에서 distillation target function $$f$$ 는 이론상으로 아무 레이어의 output mapping으로 생각할 수 있다. 하지만, distance/angle-wise losses는 teacher의 개별적인 outputs를 transfer하지 않기 때문에, 개별 output 값들 자체가 중요한 곳에 혼자 사용하는 것은 적절치 않다. Ex) softmax layer for classification

그런 경우에는, IKD 로스나 task-sepcific 로스를 함께 사용하는 것이 필요

그 밖에 대부분의 다른 경우는 RKD가 적용가능하고 효과적이었음



## 4. Experiments

3가지 태스크를 평가

- metric learning, classification, few-shot learning

- RKD-D = RKD with the distance-wise loss
- RKD-A = RKD with the angle-wise loss
- RKD-DA = RKD with two losses together
- 학습할 때, 로스들이 다른 로스들과 결합해 있는 경우 각각의 loss항에 각가의 balancing factor들을 붙여줌

RKD를 다른 KD 방법들과 비교함

- [27] FitNet
  - model은 두가지 단계로 학습
  - (1) FitNet loss와 함께 모델을 학습
  - task-specific loss와 함께 모델을 fine-tune함 (at hand?)
- [47] Attention
- [11] HKD (Hinton's)
- ++ metric learning에서는, 추가적으로 [7] Dark-Rank 도 비교 함
  - Dark-Rank = metric learning을 위해서 디자인된 KD 방법이기 때문에

비교할 때, hyperparameter의 공정한 비교를 위해 grid search를 통해 각각의 방법들 모두를 튜닝함

​	

### 4.1 Metric learning

여러 다른 task들 중에서 relational knowledge와 가장 관련있는 태스크라고 볼 수 있음

metric learning은 data examples들을 하나의 매니폴드로 projects하는 embedding model을 train하는데 목표가 있다.

- 이 manifold 두 exmaples가 의미상으로(sementically) 비슷한 경우, 서로 가까운 metric을 갖고, 다른 경우 멀리 떨어지게 함

- Metric learning aims to train an embedding model that projects data ex- amples onto a manifold where two examples are close to each other if they are semantically similar and otherwise far apart. 

- embedding 모델들은 일반적으로 image retrieval (이미지 검색)에 대해 평가하므로, 다음과 같이 data set을 고르고 validate함
  - CUB-200-2011 [40], Cars 196 [14], Stanford Online Products [21] datasets
  - Train/test splits [21]에 제시된 방식으로 함



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
- **Attention [47]**
  - original paper에 따라, Resnet의 2nd/3rd/4th block들의 output에 대해 이 방법 적용
  - Set $$\lambda_{Triplet}$$ = 1 and $$\lambda_{Attention}$$ = 50
- **FitNet [27]**
  - original paper에 따라 2 단계로 학습
    - 먼저, a model with FitNet loss를 초기화
    - model을 fine-tune함 (이 연구의 경우 Teacher = Triplet)
  - ResNet의 2,3,4번째 블록의 아웃푹과 final embedding 에 대해 이 방법 적용
- **DarkRank [7]**
  - 데이터 표본들 사이의 유사도 순위를 transfer하는 KD 방법 (metric learning을 위한 KD method)
  - 이 논문에서 2가지 로스가 제안되었는데, 여기서는 HardRank loss를 사용함
    - 연산량이 효율적, 성능이 다른거에 비해 좋기 때문
  - DarkRank loss는 teacher/student 모델의 최종 outputs에 적용함
  - 학습하는 동안, 논문에서 제시한 triplet loss와 똑같은 objective를 사용함
  - hyperparameters optimization
    - alpha = 3, beta = 3, lambda_DarkRank = lambda_Triplet = 1
    - (GridSearch on alpha 1 to 3, beta 2 to 4)
    - 원논문보다 얘네 하이퍼파라미터가 더 좋은 결과를 냄
    - 

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

- 실험세팅
  - RKD 비교 대상 - IKD, HKD, FitNet, [47] Attention
    - FitNet, Attention은 CNN의 2번째 3번째 4번째 블록들의 아웃풋에 대해 적용
    - lambda_attention = 50 세팅
    - HKD는 최종 아웃풋에 적용, 온도 타우는 4, lambda_HKD = 16
    - RKD-D와 RKD-A는 teacher/student network의 마지막 pooling layer에 적용
    - lambda_RKD-D = 25, lambda_RKD-A = 50
    - 모든 세팅의 공통으로, 마지막 로스는 CE loss로 사용
    - teacher/student모델 모두, 마지막 풀링 레이어 이후의 FC 레이어들을 제거하고, classifier로서의 역할을 수행하는 single FC layer만 달음
  - CIFAR-100, Tiny ImageNEt 데이터셋
    - CIFAR-100 = 32, 32 images with 100 object categories
      - CIFAR-100은 zero-padded 40, 40 images에 대해 랜덤 크롭해 32,32만들고 random horizontal flipping	
      - momentum 0.9, weight decay 5 * 10^(-4), mini-batch size 128, SGD optimizer
      - 200 epochs로 훈련시킴
      - learning rate는 0.1에서 시작해서 60, 120, 160 에폭마다 0.2씩 곱함
      - ResNet50을 Teacher model, VGG11 w/ batch normalization을 Student model로 채택
    - Tiny ImageNet = 64 64 with 200 classes
      - data-aug (random rotation, color jittering, horizontal flipping)
      - optimizer (SGD, mini-batch 128, momentum 0.9)
      - 300 epoch train
      - learning rate 0.1에서 시작해서 60, 120, 160, 200, 250마다 0.2씩 곱함
      - ResNet101 = teacher / ResNet18 = Student
  - ![image-20200127174449521](/Users/skcc10170/Library/Application Support/typora-user-images/image-20200127174449521.png)
  - 결과
    - RKD-DA와 HKD를 결합한 것이 성능이 가장 좋음
    - RKD 방법은 다른 KD에 대해 complementary함
      - 대부분의 경우 RKD를 붙였을 때, 성능이 향상되기 때문

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



