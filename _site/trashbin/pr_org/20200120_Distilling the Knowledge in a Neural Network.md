# Distilling the Knowledge in a Neural Network



## Abstract

- ensemble : very simple way to improve the performance

- However, using ensemble models : cumbersome & too computationally expensive when deployment
  - to a large number of users
  - the inidividual models are large NNs
  
- Caruna paper

  - C. Buciluaˇ, R. Caruana, and A. Niculescu-Mizil. Model compression. In *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD ’06, pages 535–541, New York, NY, USA, 2006. ACM.

  - compress the knowledge in an ensemble into a single model
  - single model = much easier to deploy

- We develop this approach & using a different compression technique
  - surprising results on MNIST
  - improve the acoustic model of a heavily used commercial system
  - by distilling the knowledge in an ensemble of models into a single model
  
- **<u>We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish fine-grained classes that the full models confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.</u>**



## Introduction

P1 : Analogy(?)

- Insects : larval form / adult form : optimized for different requirements, respectively
- In large-sclae machine learning, typically use similar models for the training stage / deployment stage (despite their very different requirements)
  - ex) speech and object detection
    - Training - extract structure from very large redundant datasets, not real-time, can use a huge amount of computation
- BUT) Deployment to a large number of users, has much more stringent requirements on latency and computational resources
- Analogy with insects suggests - should be willing to train cumbersome models if that makes it easier to extract structure from the data
- The cumbersome model
  - ensemble / a single very large model trained with a very strong regularizer such as dropout
- After combersome model trained -> "distillation"
  - to transfer the knowledge from the cumbersome model to a small model
  - small model's purpose : more suitable for deploymendt
- Paper : the knowledge acquired by a large ensemble of models can be transferred to a single small model!!! (demonstrate convincingly )



P2 : Realtive probabilities of incorrect answers

- A conceptual block

  - we tend to identify the knowledge : be in a trained model with the learned parameter values
  - hard to see how we can change the form of the model but keep same knowledge

- Abstract view of model's "Knowledge"

  - free from any particular instantiation
  - learned mapping from input vectors to ouput vectors

- Side effect of the learning

  - normal training objective = to maximize the average log probability of the correct answer

  - side-effect = the relative probabilities of incorrect answers

    - tell us how the cumbersome model tends to generalize

    - ex) BMW, garbage truck, carrot

      

P3 : Distillation

- the objective function reflect the true objective of the user, but models are usually trained to optimize performance on the training data when the real objective is to generalize well to new data
- Our objective : train models generalize well
  - requires information about the correct way to generalize and this information is not normally available
- Our "Distillation" can train the small model to generalize in the same way as the large model
  - cumbersome model generalizes well
    - ensemble / a large single model with a strong regularizer
  - Which one is the good one? (for A small model training)
    - trained to generalize in the same way
    - trained in the normal way



P4: "soft targets"

- **the class probabilities produced by the cumbersome model**
- an obvious way to transfer the generalization ability of the cumbersome model to a small model
- using for training the small model
  - Transfer stage (using soft targets to train the small model)
    - the same training set / a separate "transfer set"
- **<u>When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.</u>**



P5 : Our general solution "distillation"

- cubmersome model in MNIST
  - always produce the correct answer with very high confidence
  - much of the information about the learned function resides in the ratios of very small probabilities in the soft targets
    - ex) one version of 2 -> output class 3 : 10^(-6), class 7 : 10^(-9)
      - says which 2's look like 3's and which look like 7's
      - **value information** = a rich similarity structure over the data
      - **But, little influence on the CE cost function during the transfer stage(because close to zero)**
- Caruna circumvent this problem by using the logits not using probabilities
  - minimize the squared difference between logits_1 (produced by cumbersome model) and logits_2 (produced by small model)
- Our more general solution "distillation"
  - raise the temperature of the final softmax until the cumbersome model produces a suitably soft set of targets
  - use the same high temperature when training the small model to match theses soft targets
  - Caruna(matching the logits of the cumbersome model) = a special case of distillation



P6 : transfer set

transfer set

- be used to train the small model
- could consist entirely of unlabeled data or original training set
- add a small term to the objective function -> using original training set works well
  - encourages the small model to predict the true targets as well as matching the soft targets provided by the cumbersome model
- **<u>small model cannot exactly match the soft targets and erring in the direction of the correct answer turns out to be helpful</u>**





## 2 Distillation

typically, using "softamx" output layer that converts the logit, produce class probabiltiies by comparing z_i with the other logits

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200124105740736.png" alt="image-20200124105740736" style="zoom:50%;" />

- T = temperature (normally set to 1)
  - Using a higher value for T produces a softer probability distribution over classes



The simples form of distillation (how be the knowledge transfered)

- training the distilled model on a transfer set = using a soft target distribution (prouced by using the cumbersome model with a high T in its softmax)
- same high T when training the distilled model
- after trained, uses a T = 1

**<u>When the correct labels are known for all or some of the transfer set, this method can be significantly improved by also training the distilled model to produce the correct labels. One way to do this is to use the correct labels to modify the soft targets, but we found that a better way is to simply use a weighted average of two different objective functions.</u>** 

- 1st obj function = CE with soft targets
  - using the same high T in the distilled model (as was used in cumbersome model)
- 2nd obj function = CE with correct labels
  - for distilled model and using T =1 
- generally, using a condiderably lower weight on the 2nd obj func is good
- Multiply them(?) by T^2 when using both hard and soft targets
  - magnitues of the gradients produced by the soft targets scale as 1/T^2
  - relative contributions of the hard and soft targets remain roughly unchanged if the temperature used ofr distillation is changed



### 2.1 Matching logits is a special case of distillation





## 3 preliminary experiments on MNIST

a single large NN

- with 2 hidden layers of 1200 rectified linear hidden units
- 60,000 training cases
- Droput and weight-constraints
- the input images were jittered by up to two pixels in any direction
- 67 test errors

a smaller net

- with 2 hidden layers of 800 rectified linear hidden units
- no regularization -> 146 errors

temperature 20으로 생성된 large net으로부터 생성된 soft targets를 matching하는 태스크를 추가하여 정규화시킬 경우, 74 test errors를 내뱉음 -> soft targets이 상당한 부분의 knowledge를 distilled model에게 전달하는 것으로 보임,

**<u>This shows that soft targets can transfer a great deal of knowledge to the distilled model, including the knowledge about how to generalize that is learned from translated training data even though the transfer set does not contain any translations.</u>**



## 4. Experiments on speech recognition

​	



## 5. Training ensembles of specialists on very big datasets

ensemble을 훈련시킬 땐, 병렬 계산의 장점을 이용하고, test시점에 계산량을 너무 많이 필요로 할 때는 distillation을 이용해 처리할 수 있음. 그러나 개별 모델들이 너무 큰 모델들이고 데이터셋도 큰 경우, 학습할 때 계산량이 너무 초과될 수 있다는 문제가 있음

혼동하기 쉬운 특별한 부분집합에 대해서만 training하는 specialist 모델을 만들어 training을 효율적으로 할 수 있음.

The main problem with specialists that focus on making fine-grained distinctions is that they overfit very easily and we describe how this overfitting may be prevented by using soft targets.

### 5.1 JFT dataset

- 100 milion labeled images with 15000 labels
- Google's baseline model for JFT
  - 6 months using asynchronous stochastic gradient descent on a large number of cores.
  - used 2 types of parallelism
  - 이 두가지 방법을 포함하는 parallelism이 ensemble training이지만, 더 많은 core를 필요로 하므로 baseline model을 향상시키기 위한 더 빠른 방법이 필요했음

### 5.2 Specialist Models

- 클래스 수가 많은 경우, ensemble은 모든 데이터에 대해 훈련된 generalist model과 many "specialist"models를 포함시키게 한다고 볼 수 있음
  - 각각의 specialist는 (다른 종류의 버섯들)과 같은 매우 혼동가능한 클래스들의 부분집합들로부터 추린 고농축된 데이터로부터 훈련이 됨
  -  The softmax of this type of specialist can be made much smaller by combining all of the classes it does not care about into a single dustbin class.
- 오버피팅을 방지하고 low level feature를 공유하기 위해, 각각의 specialist model은 generlist model의 가중치로 초기화시킴
- 각각의 specialist 학습시 special subset과 training set의 나머지로부터 랜덤하게 뽑은 샘플 셋을 반반씩 섞어서 training
- After training, we can correct for the biased train- ing set by incrementing the logit of the dustbin class by the log of the proportion by which the specialist class is oversampled.

### 5.3 Assigning classes to specialists

specialists를 object category grouping하기 위해, full network가 자꾸 혼동하는 카테고리들에 대해 focus함

confusion matrix를 계산해서 클러스터들을 찾는 방법으로 사용해도 되지만, 우리는 true labels를 필요로 하지 않는 더욱 simpler approach를 선택

특히, 우리의 generalist model의 예측값의 covariance matrix에 clustering algorithm을 적용하여, 종종 함께 예측되는 클래스틀의 집합 S^m을 우리의 specialist models m의 targets으로서 사용함

on-line version of the K-means algorithm을 covariance matrix의 columns들에 적용했고, 합당한 클러스터들을 얻음

비슷한 결과를 내기 위해 여러번 클러스터링 알고리즘을 수행

### 5.4 Performing inference with ensembles of specialists

specialists들을 포함한 앙상블이 얼마나 성능을 내는지 확인해보고 싶었음

input image X가 주어졌을 때, top-one classification을 다음과 같은 두 가지 단계로 수행

Step 1. 각각의 test case 때, generalist model로부터 가장 높은 확률인 n개의 클래스들을 찾음. 우리 실험에서는 n=1

이 클래스들의 집합을 k라고 부름

Step 2. 모든 specialist models를 가져와서, S^m과 k의 non-empty intersection을 active set of specialists A_k를 생각. 모든 클래스들에 대하여 probability distribution q를 찾을 수 있음



## 6. Soft Targets as Regularizers

a single hard target에서 녹일 수 없는 수많은 유용한 정보들을 soft targets로 전달해줄 수 있었다.




















