Zero-Shot Knowledge Distillation in Deep Networks 을 읽기 위해
(https://arxiv.org/pdf/1905.08114.pdf)

Distilling the Knowledge in a Neural Network를 먼저 읽음
(https://arxiv.org/pdf/1503.02531.pdf)

#### Abstract

- 머신러닝 알고리즘 성능을 향상시키기 위한 가장 간단한 방법 중에 앙상블이 있음
- 그러나 앙상블을 통해 예측하는건, 매우 cumbersome하고 computationally expensive하다. (특히, 개별 신경망 모델들이 너무 크면서, 많은 유저들에게 deployment를
허락시킬때) = 즉, 큰 모델을 사용자들에게 제공하려고 할 때 앙상블을 제공하는 것은 성가시고, 연산스펙이 너무 크다.
- 과거에 앙상블을 단일모델로 knowledge compression 가능하다는 것을 보인 사례가 있음
(C. Buciluˇa, R. Caruana, and A. Niculescu-Mizil. Model compression. In Proceedings of the
12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD
’06, pages 535–541, New York, NY, USA, 2006. ACM.)
- 위 papeor에서는 저런 kc를 하는 이유가 바로 much easier to deploy하기 위해서 단일 모델로 압축시키기 위함임.

- 그리고 이 paper에서는 다른 compression 기술을 사용해 위 paper에 대한 접근보다 더 발전시킴.
- MNIST에 대해 놀라운 결과들을 얻었으며, 모델들의 앙상블 안에 있는 knowledge를 단일 모델로 distilling 시킴으로써
the acoustic model of a hevaily used commerical system을 상당히 향상시킬 수 있다는 것을 보임.

??
- We also introduce a new type of ensemble composed of one or more full models and many
specialist models which learn to distinguish fine-grained classes that the full models confuse. 
- Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.






#### 1 Introduction

Many insects have a larval form that is optimized for extracting energy and nutrients from the environment
and a completely different adult form that is optimized for the very different requirements
of traveling and reproduction.

In large-scale machine learning, we typically use very similar models
for the training stage and the deployment stage despite their very different requirements: For tasks
like speech and object recognition, training must extract structure from very large, highly redundant
datasets but it does not need to operate in real time and it can use a huge amount of computation.

- 많은 유저에게 Deploy하는 것은 latency와 computational resources이라는 더욱 엄격한 요구사항을 갖는다.
- 

The analogy with insects suggests that we should be willing to train
very cumbersome models if that makes it easier to extract structure from the data. The cumbersome
model could be an ensemble of separately trained models or a single very large model trained with
a very strong regularizer such as dropout [9]. Once the cumbersome model has been trained, we
can then use a different kind of training, which we call “distillation” to transfer the knowledge from
the cumbersome model to a small model that is more suitable for deployment. A version of this
strategy has already been pioneered by Rich Caruana and his collaborators [1]. In their important
paper they demonstrate convincingly that the knowledge acquired by a large ensemble of models
can be transferred to a single small model.


A conceptual block that may have prevented more investigation of this very promising approach is
that we tend to identify the knowledge in a trained model with the learned parameter values and this
makes it hard to see how we can change the form of the model but keep the same knowledge. A more
abstract view of the knowledge, that frees it from any particular instantiation, is that it is a learned
∗Also affiliated with the University of Toronto and the Canadian Institute for Advanced Research.
†Equal contribution.
1
mapping from input vectors to output vectors. For cumbersome models that learn to discriminate
between a large number of classes, the normal training objective is to maximize the average log
probability of the correct answer, but a side-effect of the learning is that the trained model assigns
probabilities to all of the incorrect answers and even when these probabilities are very small, some
of them are much larger than others. The relative probabilities of incorrect answers tell us a lot about
how the cumbersome model tends to generalize. An image of a BMW, for example, may only have
a very small chance of being mistaken for a garbage truck, but that mistake is still many times more
probable than mistaking it for a carrot.
It is generally accepted that the objective function used for training should reflect the true objective
of the user as closely as possible. Despite this, models are usually trained to optimize performance
on the training data when the real objective is to generalize well to new data. It would clearly
be better to train models to generalize well, but this requires information about the correct way to
generalize and this information is not normally available. When we are distilling the knowledge
from a large model into a small one, however, we can train the small model to generalize in the same
way as the large model. If the cumbersome model generalizes well because, for example, it is the
average of a large ensemble of different models, a small model trained to generalize in the same way
will typically do much better on test data than a small model that is trained in the normal way on the
same training set as was used to train the ensemble.
An obvious way to transfer the generalization ability of the cumbersome model to a small model is
to use the class probabilities produced by the cumbersome model as “soft targets” for training the
small model. For this transfer stage, we could use the same training set or a separate “transfer” set.
When the cumbersome model is a large ensemble of simpler models, we can use an arithmetic or
geometric mean of their individual predictive distributions as the soft targets. When the soft targets
have high entropy, they provide much more information per training case than hard targets and much
less variance in the gradient between training cases, so the small model can often be trained on much
less data than the original cumbersome model and using a much higher learning rate.
For tasks like MNIST in which the cumbersome model almost always produces the correct answer
with very high confidence, much of the information about the learned function resides in the ratios
of very small probabilities in the soft targets. For example, one version of a 2 may be given a
probability of 10−6 of being a 3 and 10−9 of being a 7 whereas for another version it may be the
other way around. This is valuable information that defines a rich similarity structure over the data
(i. e. it says which 2’s look like 3’s and which look like 7’s) but it has very little influence on the
cross-entropy cost function during the transfer stage because the probabilities are so close to zero.
Caruana and his collaborators circumvent this problem by using the logits (the inputs to the final
softmax) rather than the probabilities produced by the softmax as the targets for learning the small
model and they minimize the squared difference between the logits produced by the cumbersome
model and the logits produced by the small model. Our more general solution, called “distillation”,
is to raise the temperature of the final softmax until the cumbersome model produces a suitably soft
set of targets. We then use the same high temperature when training the small model to match these
soft targets. We show later that matching the logits of the cumbersome model is actually a special
case of distillation.
The transfer set that is used to train the small model could consist entirely of unlabeled data [1]
or we could use the original training set. We have found that using the original training set works
well, especially if we add a small term to the objective function that encourages the small model
to predict the true targets as well as matching the soft targets provided by the cumbersome model.
Typically, the small model cannot exactly match the soft targets and erring in the direction of the
correct answer turns out to be helpful.


#### 2 Distillation

- 일반적으로 신경망은 "softmax" output layer를 통해 클래스에 대한 확률값을 내뱉는데, 이는 로짓값 z_i를 





