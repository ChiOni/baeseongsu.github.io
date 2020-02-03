---
layout: post
title:  "Paper Review : Relational Knowledge Distillation"
date:   2020-01-28 22:22
lastmod : 2020-01-31 14:44
sitemap :
  changefreq : daily
  priority : 1.0
categories: [Paper Review, Deep Learning, Knowledge Transfer]
use_math: true
---

<br/>

<br/>

<br/>

안녕하세요. 배성수입니다.

오늘 리뷰할 논문은 CVPR 2019에 Accepted Paper인 **"Relational Knowledge Distillation"** 입니다.

최근 딥러닝 스터디를 하며 Knowledge Distillation(KD) 관련 논문 2편을 읽었습니다.

1. Distilling the Knowledge in a Neural Network, NIPS 2014 Deep Learning Workshop
2. Zero-Shot Knowledge Distillation in Deep Networks, ICML 2019

기존 KD에 대한 여러 방법을 잘 알지 못하지만, 위에 나열된 2편의 KD논문에 비해 오늘 리뷰할 논문이 좀 더 명확하고 깔끔하다고 느껴졌습니다. 특히 핵심 아이디어가 뚜렷하고, 뒷받침하는 실험들의 세팅이 탄탄하다는 느낌을 받았습니다. 저자는 Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho 입니다. 재밌게 잘 읽었습니다. 감사합니다.

<br/>

<br/>

<br/>

---

<br/>

# <center> Relational Knowledge Distillation, CVPR 2019 </center>

<br/>

<br/>

<br/>

## 1. Introduction 

최근 Computer Vision이나 Artificial Intelligence 연구에선 많은 연상량과 메모리를 필요로 하는 모델들이 자주 등장합니다. 이러한 물리적 부담을 줄이기 위한 방법 중 하나로 모델의 지식(knowledge)을 전달(transfer)하는 방법이 있습니다. 이러한 Knowledge Transfer에 있어서, 가장 핵심이 되는 2가지 질문이 있습니다. 바로 "학습된 모델에 들어있는 지식은 무엇으로 구성되어 있는가?"와 "그 지식을 다른 모델로 어떻게 전달할 것인가?"입니다.

예를 들어, Hinton 교수님의 KD 방법에서 지식이란, 입력으로부터 출력까지 학습된 매핑함수를 의미합니다. 또한, Teacher 모델의 Soft Targets을 이용해 지식을 전달하는 방식을 취하고 있습니다. 이처럼 두 가지 질문으로부터 KD 연구는 진행된다고 해도 과언이 아닙니다.

특히, 이 논문은 언어적 구조주의(linguistic structuralism) 관점에서 KD가 중요하게 생각해야할 점들에 대해 설명하고 있습니다. 언어도 일종의 기호학 체계로 볼 수 있고, 언어에 녹아든 기호의 의미는 큰 체계 속에서 다른 기호들과 어떤 관계를 맺고 있는지가 중요하다는 뜻입니다. (자세한 내용은 논문과 소쉬르의 구조주의 언어학을 참조해주세요.)

<br/>

<center>
  <figure>
    <img data-action="zoom" src='{{ "/assets/img/pr/rkd_figure_1.png" | relative_url }}' alt='absolute' width="60%" height="60%">
    <figcaption> 
      <span style="font-size:10pt"> 위 그림을 통해 기존의 KD와 RKD의 차이를 명확히 볼 수 있습니다. </span> 
    </figcaption>
  </figure>
</center>



<br/>

그렇다면, 앞서 말한 2가지 질문에 대해 이 논문은 어떻게 답할 수 있을까요? 먼저, "지식이라는 것은 (학습된 상황일 때) 개별적인 representation보다 그들의 관계에 의해 더 잘 표현된다"라는 것입니다. 그에 따라, "개별적인 output보다 output들의 구조적 관계를 전달하는 방식으로 지식을 전달하는 방식으로 접근할 것이다"라고 설명하고 있습니다. 이러한 점에서 기존 KD를 일반화할 수 있는 RKD라는 것이 탄생하게 되었고, 지식전달능력도 상당히 우수하다고 보였습니다. 결국, 지식은 관계 속에 녹아들어 있고, RKD는 이러한 지식을 전파하는데 있어 효과적인 방법이라고 설명합니다.

<br/>

<br/>

<br/>

## 2. Related Work

한 모델의 지식(Knowledge)을 다른 모델로 전달(Transfer)하는 연구는 꽤 오랫동안 지속됐다고 합니다. 처음으로, Breiman and Shang이 트리 기반의 model compression을 통해 지식을 전달하는 방법을 제안합니다. 그 이후로, 신경망 분야의 model compression이 등장했고, Hinton 교수님은 soft targets라는 컨셉을 이용하여 지식 증류(Knowledge Distillation, a.k.a KD)라는 네이밍을 탄생시킵니다. 최근에는 HKD(Hinton's KD)를 이은 후속 연구들뿐만 아니라 기존 접근과 다른 방식의 연구가 진행되고 있으며, 지도학습을 넘어 준지도학습/비지도학습 영역에서의 KD, 태스크에 특화된 KD 등에 대한 연구가 진행되고 있습니다.

다양한 KD 연구흐름 속에서 Chen의 연구(Darkrank: Accelerating deep metric learning via cross sample similarities transfer)가 rank loss를 사용해 similarities를 transfer하는 metric learning 기반의 KD라는 점에서 이 연구와 유사성이 어느정도 있습니다. 그러나, Chen의 연구는 metric learning에만 제한되어 있고, 본 연구는 다양한 테스크에 적용가능한 general framework라는 차이점이 있습니다. 게다가, metric learning task에서 Chen의 KD방법보다 성능이 더 좋습니다.

<br/>

<br/>

<br/>

## 3. Our Approach

보편적으로 사용해 온 지식증류(Knowledge Distillation)를 살펴본 뒤, 논문에서 제안한 관계형 지식증류(RKD, Relational Knowledge Distillation)의 핵심 개념에 대해 살펴보겠습니다. 또한 RKD에서 사용되는 손실함수로서, 간단하면서도 효과적인 두 가지 증류 손실함수(distillation losses)에 대해 자세히 알아보겠습니다.

<br/>

<br/>

### 3.0 Notation

먼저, 논문에서 사용하는 Notation에 대해 알아보겠습니다.

&nbsp; (1) 주어진 Teacher model $T$ , Student model $S$ 이 일반적으로 Deep Nueral Network라고 형태라고 생각했을 때, 해당 모델의 mapping function을 각각 $f_T$ 와 $f_S$ 라고 표시합니다. $f$ 라는 함수는 신경망의 어떤 층이든 상관없이 그 층의 출력으로 정의될 수 있으나, 일반적으로 최종 출력을 의미하는 경우가 많습니다.

&nbsp; (2) 서로 다른 data examples의 $N$-튜플 형태를 $\chi^{N}$이라고 표기합니다. 예를 들면, $\chi^{2}$ 라면 $\\{ ( x_i, x_j ) \, \| \,  i \neq j  \\}$ 와 같은 distinct pair set, $\chi^3$ 라면 $\\{ (x_i, x_j, x_k) \, \| \,  i \neq j \neq  k \\}$ 와 같은 distinct triplet set로 볼 수 있습니다.

<br/>

<br/>

### 3.1 Conventional KD

일반적인 KD 방법은 다음과 같은 목적함수를 갖습니다.
<br/>

<br/>

$$\mathcal{L}_{\text{IKD}} = \sum_{x_i \in \chi}{l( f_T(x_i), f_S(x_i) )}$$

<br/>즉, Teacher와 Student 모델로 나온 각각의 output mapping을 비슷하게 만들도록 학습합니다. 본 논문에서는, 이러한 종류의 지식 전달 방법을 개별적인 Teacher 모델의 출력값을 Student 모델에게 전해준다는 점에서 IKD(Individual Knowledge Distillation)라고 부르고 있습니다.

<br/>

<br/>

### 3.2 Relational KD

<br/>

<center>
	<img data-action="zoom" src='{{ "/assets/img/pr/rkd_figure_2.png" | relative_url }}' alt='a' width="100%" height="100%">
</center>

<br/>

RKD의 목표는 Teacher 모델의 output representation에서 data examples의 mutual relations를 이용해 구조적 지식(Structural Knowledge)를 전달하는 것으로 볼 수 있습니다. 따라서 기존 KD가 지식을 전달하는 방식과 달리, RKD는 $n$-tuple 형태의 data examples에 대한 relational potential $\psi$ 를 계산하고, 그 포텐셜 값을 통해 지식을 전달합니다.

<br/>

$$\mathcal{L}_{\text{RKD}} = \sum_{(x_{i},.., x_{j})\in\chi^{2}}{l_{\delta}{(\psi{(t_i,..,t_j)}, \psi{(s_i,..,s_j)})}}$$

$$ \text{where } t_i =  f_{T}(x_{i}), \; s_{i} = f_{S}(x_{i}), \; \psi : \text{relational potential function}$$ 

<br/>

바로 이 relational potential 덕분에 high-order properties 형태의 지식을 전달가능하게 됩니다. (= High-order propertiy is invariant to lower- order properties, even regardless of difference in output dimensions between the teacher and the student.)

<br/>

<u>Q) 그렇다면 RKD가 IKD의 일반화라고 볼 수 있을까요?</u>

네, RKD는 relational potential 관점에서 IKD를 확장시킨 것이라고 볼 수 있습니다. RKD의 relation이 unary일 경우, 즉 튜플의 개수가 1일 경우에 대해 potential function $$\psi$$ 가 identity라고 본다면 IKD와 동일하기 떄문입니다.

이처럼, RKD에서 relational potential function의 역할은 굉장히 중요합니다. 이 함수를 어떻게 정의하냐에 따라 RKD의 결과와 효율성이 달려있기 때문입니다. 예를 들어, high-order potential의 경우 higher-level struture을 잘 잡아내지만 computationally expensive합니다. 그래서 본 논문에서는 효과적인 2개의 potential function을 소개하며, 이와 연관된 RKD의 손실함수에 대해 제안합니다. 

<br/>

<br/>

#### 3.2.1 Distance-wise distillation loss

$\psi_{\text{D}}$ 라는 거리 기반의 포텐셜 함수(distance-wise potential function)를 $\psi_{\text{D}} (t_{i},t_{j}) = \frac{1}{\mu}{\|\|t_{i}-t_{j}\|\|}_2$ 라고 정의합니다. 즉, 한 쌍을 이루는 두 개의 데이터 샘플이 신경망을 통해 output representation space에 놓여질 때, 그들간의 유클리디안 거리를 계산하는 함수라고 보시면 됩니다. 여기서 $\mu$ 는 거리함수의 normalization factor 입니다. 그렇다면, 이 $\mu$ 는 어떻게 정하는 것이 좋을까요?

<br/>

논문의 핵심 아이디어가 결국 관계성에 있기 때문에, 다른 쌍들과 비교하여 상대적 거리를 구할 수 있게 $\mu$ 를 선택하게 됩니다. 따라서 쌍으로 구성된 미니배치인 $\chi^{2}$ 에서 나온 각각의 페어 데이터의 평균 거리로 계산하게 됩니다.

<br/>

$$  {\mu = \frac{1}{\| \chi^2 \|} \sum_{(x_i, x_j) \in \chi^{2} } {\| t_i-t_j \|_2}}  $$

<center>
  <span style="font-size:10pt"> Normalization Factor $\mu$
  </span>
</center>

<br/>

만약 $\mu$ 와 같은 factor가 존재하지 않는다면, Teacher 모델의 dimension 일반적으로 더 크기 때문에 Teacher 모델과 Student 모델 사이의 거리 scale 차이가 발생하게 됩니다. 따라서 논문에서는 $\mu$ 를 사용하여 $\psi_{\text{D}}$ 라는 포텐셜 함수가 결국 distance-wise potentials를 잘 반영할 수 있도록 합니다. 실제로 $\mu$ 라는 factor로 인해 학습이 더 안정적이고 빠르게 수렴하는 것을 관찰했다고 합니다.

<br/>

앞에서 살펴본 $\psi_{\text{D}}$ 를 사용해 거리기반 증류 손실함수(Distance-wise distillation loss)는 다음과 같이 정의합니다.

<br/>

$$\mathcal{L}_{\text{RKD-D}} = \sum_{(x_{i}, x_{j})\in\chi^{2}}{l_{\delta}{(\psi_{\text{D}}{(t_i,t_j)}, \psi_{\text{D}}{(s_i,s_j)})}}$$

<center>
  <span style="font-size:10pt"> 여기서 $l_{\delta}$는 Huber loss를 의미합니다. (외부에선 MAE, 이상치 덜 민감 - 내부에선 세밀하게 MSE) 
  </span>
</center>

<br/>

결국 이 손실함수는 모델의 output representation spaces 내 상대적 거리를 비슷하게 만들도록 해서, 쌍으로 이루어진 데이터가 있을 때 그들의 관계(relationships)들을 Student 모델로 전달(transfer)하는 역할을 하게 됩니다. Student의 output이 직접적으로 Teacher 모델의 output 값을 맞추도록 강요하는 것이 아니라, output이 놓여지는 공간의 거리구조에 초점을 맞추도록 한다는 것이 기존 KD와의 차이점이라고 할 수 있습니다.

<br/>

<br/>

#### 3.2.2 Angle-wise distillation loss

앞에서 살펴본 $\psi_{\text{D}}$ 가 pair로 작동하는 방식이었다면, 하나의 차원이 더 늘어난 triplet은 어떤 방식으로 작동할까요? 세 쌍이 주어진 경우, output representation space에서 생기는 angle에 대한 metric을 생각해볼 수 있습니다. 따라서, angle-wise potential function $\psi_{\text{A}}$ 는 다음과 같이 정의할 수 있습니다.

<br/>


$$ \psi_{\text{A}}{(t_i, t_j, t_k)} = cos \angle{t_{i}t_{j}t_{k}} = \langle \mathbf{e}^{ij}, \mathbf{e}^{kj} \rangle$$ 

$$\text{where } \mathbf{e}^{ij} = \frac{t_i-t_j}{\|t_i-t_j\|_2}, \mathbf{e}^{kj} = \frac{t_k-t_j}{\|t_k-t_j\|_2}.$$ 

<br/>

동일한 방식으로 각도 기반의 증류 손실함수(Angle-wise distillation loss)를 생각한다면, 다음과 같이 표기할 수 있습니다.

<br/>

$$\mathcal{L}_{\text{RKD-A}} = \sum_{(x_i,x_j,x_k)\in\chi^{3}}{l_{\delta}{(\psi_{\text{A}}{(t_i,t_j,t_k)}, \psi_{\text{A}}{(s_i,s_j,s_k)})}}$$

<br/>

기존의 distance-wise 보다 angle-wise가 더 higher-order property이기 때문에, 학습 과정에서 관계형 정보를 Student 모델에게 더욱 효과적이고 유연하게 전달할 수 있습니다. 실제 실험에서, angle-wise loss가 종종 더 빠르고 수렴하고, 좋은 성능을 보이는 것을 관찰했다고 합니다.

<br/>

<br/>

#### 3.2.3 Training with RKD

학습과정에서 제안된 RKD 손실함수는 단독으로 사용할 수도 있고,  task에 특화된 손실함수와 함께 사용할 수도 있습니다. 따라서 전체적인 목적함수(objective)를 수식으로 표현하면 아래와 같은 형태가 됩니다.

<br/>

$$\mathcal{L}_{\text{task}} + \lambda_{\text{KD}} \cdot \mathcal{L}_{\text{KD}}$$

<br/>

추가적으로 이 논문에서 RKD에서 제안된 증류 손실함수들을 구할 때는, tuple sampling을 미니배치 속 표본에 대해 가능한 모든 조합으로 구성한다고 합니다. $\lambda$ 와 같은 balancing factor는 모델의 하이퍼파라미터로서 작동합니다.

<br/>

<br/>

#### 3.2.4 Distillation target layer

RKD에서 distillation target function $f$ 는 이론적으로 어떤 레이어의 output mapping을 쓰든 상관이 없습니다. 하지만, distance/angle-wise losses는 Teacher 모델의 개별적인 출력값에 대한 지식을 전달해주지 않기 때문에 개별 output값들 자체가 중요한 곳에 혼자 덜렁 사용하는 것은 적절치 않습니다. 그런 경우에는 IKD 손실함수나 task-specific 손실함수를 사용하는 것이 필요합니다. 그 밖에 대부분의 다른 경우는 RKD가 적용가능하고 효과적인 성능을 보인다고 합니다.

<br/>

<br/>

<br/>

## 4. Experiments

Metric learning, Classification, Few-shot learning 이라는 3가지 태스크에 대해 실험을 진행했습니다. 기존의 RKD를 사용한 손실함수에 따라 RKD-D, RKD-A, RKD-DA 로 구분하고, 다른 손실함수와 결합해서 사용할 경우 항상 각 손실함수의 조정계수(balancing factor)를 고려했다고 합니다. 각 태스크에 대하여 RKD를 FitNet, Attention, HKD(Hinton's KD), *Dark-Rank 등과 비교했고, 하이퍼파라미터의 공정한 비교를 위해 grid search로 최적화했습니다.

<span style="font-size:10pt"> *Dark-Rank = 데이터 사이의 유사도 순위를 transfer하는, metric learning에 적합한 KD 방법 (metric learning task에서만 사용) </span>

<br/>

<br/>

### 4.1 Metric learning

Metric learning은 데이터들을 하나의 manifold로 embedding하여 그들 사이의 similarity를 잘 표현하도록 하는 학습 방법입니다. 사실, relational knowledge 측면에서는 가장 관련 있는 태스크라고 볼 수 있습니다.

Teacher/Student 모델의 세팅과 학습 방법은 논문에 잘 설명되어 있으니 참고하시면 됩니다. 학습이 다 끝나고 평가할 때는 query image에 대해 나온 결과를 top K-nn을 통해 Image retrieval하도록 합니다. Recall@k 값을 통해 Triplet, FitNet, Attention, DarkRank, RKD에 대해 평가합니다.

Baseline으로 사용한 Triplet과 비교하여, RKD는 L2-정규화가 필요 없다고 말하고 있습니다. RKD에서 사용하는 손실함수들은 embedding points 사이의 거리가 가지는 범위에 영향을 받지 않기 때문입니다. 또한, 이로 인해 RKD가 margin이나 triplet sampling parameters와 같은 민감한 파라미터들이 필요없다는 장점이 있습니다.

다음은 Triplet, FitNet, Attention, DarkRank를 RKD와 비교한 결과입니다.



<center>
  <img data-action="zoom" src='{{ "/assets/img/pr/rkd_table_1.png" | relative_url }}' alt='a' width="100%" height="100%">
</center>

<br/>

<br/>

#### 4.1.1 Distillation to smaller networks

Baseline과 비교했을 때, RKD는 임베딩 차원에 덜 영향을 받습니다. 또한, RKD는 임베딩 공간을 잘 활용하기 때문에 다른 방법과 달리 L2-정규화 없이 학습시키는 것이 성능이 더 좋습니다. 또한, Cars 196 데이터셋에 대해 RKD의 Student 모델이 Teacher 모델을 뛰어넘는 성능을 보이고 있습니다.

<br/>

<br/>

#### 4.1.2 Self-distillation

사자성어로 청출어람이라고 하듯, Student 모델이 Teacher 모델보다 훨씬 성능이 좋은 경우를 관찰할 수 있습니다. 따라서, RKD가 self-distillation을 통해 성능을 향상시킬 수 있는지 알아봤습니다. 실험은 RKD-DA로 전 세대를 학습하는 방식으로 진행했습니다. (이전 세대의 Student 모델을 Teacher 모델로 사용)

<center>
  <img data-action="zoom" src='{{ "/assets/img/pr/rkd_table_2.png" | relative_url }}' alt='a' width="60%" height="60%">
</center>

실험결과, self-distillation을 통해 성능이 향상된 것을 확인했고 한 번 진행하는 것이 가장 효과적이었습니다.

<br/>

<br/>

#### 4.1.3 Comparison with state-of-the art models

최근 SOTA 방법들이 대부분  GoogLeNet을 백본으로 사용했기 때문에 공정한 비교를 위해 Student 모델을 GoogLeNet과 ResNet으로 학습시켰고, 모두 같은 embedding size를 사용했습니다.

<center>
  <img data-action="zoom" src='{{ "/assets/img/pr/rkd_table_3.png" | relative_url }}' alt='a' width="100%" height="100%">
</center>

<br/>

<br/>

#### 4.1.4 Discussion

1. <u>RKD는 L2-정규화가 없을 때 성능이 더 좋다.</u>

   RKD는 L2-정규화 없이도 안정적으로 학습된다는 장점이 있습니다. 이는 임베딩한 모델의 output points를 대응시키는 방식으로부터 차이가 발생하게 됩니다. L2-norm을 통해 output points를 unit-hypersphere의 표면으로 대응시키는 반면, RKD는 Student 모델의 임베딩 공간을 모두 활용할 수 있게 되기 때문입니다. 특히 Dark-Rank와 비교했을 때, Dark-Rank는 L2-정규화없이는 성능이 너무 떨어지게 됩니다. (이는 Dark-Rank가 Triplet loss기반이기 때문이고, Triplet loss는 L2-norm이 없다면 굉장히 취약하기 때문)

2. <u>Student 모델은 가끔 Teacher 모델보다 성능이 더 좋을 때가 있다.</u>

   사자성어로 청출어람이라고 표현할 수 있는 이 현상은 바로 KD에서 생성하는 soft output 혹은 continous target labels(ex, distance or angle)들이 추가적인 유용한 정보를 담고 있기 때문에 학습시 성능이 좋아질 때가 있다고 설명하고 있습니다. Triplet loss와 같은 기존 Loss의 경우 들어가는 binary ground-truth labels들은 그러한 유용한 정보를 담고 있지 않은 것을 알 수 있습니다.

3. <u>다양한 데이터셋에 대한 일반화 능력을 희생하여, 성능을 높이다.</u>

   <center>
     <img data-action="zoom" src='{{ "/assets/img/pr/rkd_figure_3.png" | relative_url }}' alt='a' width="50%" height="50%">
   </center>
   
   
   
   Cars 196을 학습 도메인 데이터셋으로 설정한 뒤 3가지 데이터셋에 대해 domain adaptation을 적용해본 결과입니다. 이를 통해, RKD가 성능이 잘 나오는 이유가 다른 도메인에 대해 일반화하는 능력을 희생시켜, 학습 도메인에 대해 모델을 적합시키는 것을 확인할 수 있었습니다.
   

<br/>

<br/>

### 4.2 Image classification

사용한 데이터셋은 CIFAR-100과 Tiny ImageNet이고, RKD와 비교 대상으로 IKD, HKD, FitNet, Attention을 사용했습니다. 모두 cross-entropy loss를 포함해 목적함수를 세팅했고, ResNet과 VGG를 기반으로 하여 Teacher/Student 모델을 구성했습니다.

<center>
  <img data-action="zoom" src='{{ "/assets/img/pr/rkd_table_4.png" | relative_url }}' alt='a' width="60%" height="60%">
</center>



표에서 볼 수 있듯이, RKD-DA와 HKD를 함께 사용한 방법이 가장 성능이 좋습니다. 또한, RKD-DA와 결합했을 때 대부분의 경우 성능이 향상됩니다.

<br/>

<br/>

### 4.3 Few-shot learning

Few-shot classification에서 표준 밴치마크인 Omniglot와 *mini*ImageNet을 사용했고, 특히 prototypical networks를 사용해 RKD를 평가했습니다. 아래의 표를 통해, RKD가 Teacher를 뛰어넘는 Student 모델의 성능을 보여주고 있고 성능이 FitNet, Attention에 비해 좋은 것을 확인할 수 있습니다.

<center>
  <img data-action="zoom" src='{{ "/assets/img/pr/rkd_table_5,6.png" | relative_url }}' alt='a' width="50%" height="50%">
</center>



<br/>

<br/>

## 5. Conclusion

이 논문을 통해, RKD가 데이터의 mutual relations을 사용하여 지식을 효과적으로 전달하는 것을 확인했습니다. 특히 metric learning에서는 Teacher 모델보다 Student 모델이 성능이 더 잘 나올 때도 있었죠. 또한, 거리와 각도를 기반으로 하는 두 가지 RKD 손실함수를 통해 다양한 task들을 효과적으로 수행했습니다. 사실 이 두 가지의 경우를 넘어, RKD는 다양한 task-specific 손실함수를 설정할 수 있게 합니다.

마지막으로, high-order relations의 지식을 전달할 수 있는 연구 영역의 문을 RKD가 열었다고 말하며 논문을 마치고 있습니다.

<br/>

<br/>

<br/>

---



## 99. 나의 생각

- 





























