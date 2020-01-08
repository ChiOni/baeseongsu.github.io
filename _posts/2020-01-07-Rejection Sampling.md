---
layout: post
title:  "Rejection Sampling"
date:   2020-01-07 23:00
categories: Simulation
use_math: true
---
수치해석과 전산통계 분야에서 기각샘플링은 특정한 분포로부터 샘플링을 하고 싶을 때 사용하는 기본적인 방법입니다. 일반적으로 뽑힌 샘플에 대해 "채택"과 "기각" 둘 중 하나를 수행하기 때문에 "Accept-Reject Algorithm"이라고도 부릅니다. 또한, 이 알고리즘은 exact simulation method로 알려져 있습니다. (문제의 optimal solution을 항상 구해주는 method/algorithm으로 생각하면 될듯 합니다.)

### Concept
- 샘플을 쉽게 뽑을 수 있는 **proposal distribution**인 $$q(x)$$를 사용하여 **target distribution**인 $$p(x)$$의 분포를 따르는 샘플을 뽑을 수 있는 방법

### Procedure
- 샘플링하고 싶은 대상, 타겟 분포인 $$p(x)$$ 존재
- 샘플링이 가능한 대상, 제안 분포인 $$q(x)$$를 선택
- 모든 $$x$$에 대하여 $$ p(x) \leq Mq(x) $$를 만족하는 $$M$$이 존재한다고 가정

  1. uniform distribution $$ U(0,1) $$와 proposal distribution $$q(x)$$에서 각각 $$u$$, $$x^*$$을 샘플링.
  2. $$ u \leq \frac{p(x^{*})}{Mq(x^{*})} $$ 일 때, 채택(accept)
  3. 반대의 경우, 기각(reject)하고 다시 1번 과정 수행

### Idea
- 당연1) M이 아슬아슬하게 $$p(x)$$를 덮을 수 있으면 버려지는 샘플들이 적어지고 효율적임
- 당연2) 제안 분포 $$q(x)$$가 $$p(x)$$와 유사할수록 좋음
  - $$p(accept)$$ : 채택율
    - $$$M$$ 에 반비례함

### reference
- https://en.wikipedia.org/wiki/Rejection_sampling
- https://www.stats.ox.ac.uk/~teh/teaching/simulation/slides.pdf
  - Inversion Method
  - Importance sampling
  - Markov Chain Monte Carlo
  - Metropolis-Hastings
- https://www2.cs.duke.edu/courses/fall17/compsci330/lecture9note.pdf
  - rejection sampling
  - Monte-Carlo Method
- https://datascienceschool.net/view-notebook/ea4584bde04140368950ee50ef038833/
- https://untitledtblog.tistory.com/134
- http://www.stat.ucla.edu/~dinov/courses_students.dir/04/Spring/Stat233.dir/STAT233_notes.dir/RejectionSampling.pdf



---

### Proposition
$$\Omega$$ 상의 모든 $$x$$에 대하여 $$p(x)/q(x) \leq M$$인 constant $$M$$을 구할 수 있다고 가정하자. 그러면 다음의 'Rejection' 알고리즘은 $$X \sim p$$ 를 반환한다.

### Proof
$$Pr(X=x)=\sum_{n=1}^{\infty} Pr(reject\ n-1\ times,\ draw\ Y=x\ and\ accept\ it)$$

$$=\sum_{n=1}^{\infty} Pr(reject\ Y)^{n-1}\ Pr(draw\ Y=x\ and\ accept\ it)$$  

$$(\because Simulation\ Procedure\ = indepedent\ trial)$$  

$$1)\ Pr(reject\ Y)=Pr(draw\ Y=x\ and\ accept\ it)$$  

$$=Pr(draw\ Y=x)\ Pr(accept\ Y|Y=x)$$  

$$=q(x)\ Pr(U \leq \frac{p(Y)}{Mq(Y)}|Y=x)$$  

$$=q(x) \times \frac{p(x)}{Mq(x)}$$  

$$=\frac{p(x)}{M}$$  

$$2)\ Pr(draw\ Y=x\ and\ accept\ it) = Pr(draw\ Y=x)\ Pr(accept\ Y|Y=x)$$  
