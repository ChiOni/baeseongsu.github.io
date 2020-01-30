Preliminary

- Rejection sampling
  - https://untitledtblog.tistory.com/134
  - pdf에서 sampling하는 방법이 없음
  - proposoal distribution에 target distirbution보다 upper bound에 있어야함, envelope
    - pdf의 sum이 1인 것.
  - proposal distribution -> computationally problem in sample procedure
  - 과정을 비디오로 만들면?
- importance sampling 공부할 것, 두어개 더 있음
  - https://untitledtblog.tistory.com/135?category=823331
- Gaussian distribution - sampling 하는 방법
  - pdf analytic, cdf, cdf inverse analytic
  - gaussian은 가능. Analytic closed form possible
- uniform으로부터 sampling할 수 있는 dist. vs. 없는 dist.
  - 없는쪽에 있는 dist.
- 2stage를 통해 나온 p.d가 수식으로 유도가능



GAN

- Assumption
  - a1 ; support가 같은 상태로 rejection sampling써야함 (정의역이 같은가?)
  - a2 ; 
- Practical scheme
  - ill-defined $$D^*$$
  - finite size of dataset
  - high dimensional target distribution
    - Gibbing sampling / metropolis heistanc ? 공부
    - Why sigmoid? nonlinear efect
    - rejection sampling에서 해결하는 방식이 여기서처럼 되는가?
- 왜 중간에 나왔느냐 토이데이터셋에서
  - GAN의 식에서 비율을 minimize하기 때문



CAM기반 distrib. 가 생성된 이미지와 실 이미지 의 차이를 통해 구분

아니면, 이미지 생성시 CAM에 대한 정보를 반영



---



Paired pixel-level

unpaired setting -> latent vector(segmentation purposed network)

무언가 domain adaption은 source/ target 딱딱 나누어놓고 하지만,  실제로 혼재된 상황이 많을듯











