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











---

글또 4기를 지원하는 이유?

- 메모하는 습관이 중요다는건 누구나 아는 사실이지만, 우리는 <메모의 중요성>이라는 자기계발서를 보며 '역시 '메모가 중요하지. 나도 해보자.'라는 다짐을 합니다. 이처럼 사람은 동기를 부여하고 자극하는 것이 굉장히 중요하다고 생각합니다. 이처럼, 저도 "글또 4기"에 참여해 글을 작성하는 습관을 기르고 싶었습니다. (마침 글또 4기를 알기 직전에, 논문리뷰 및 정보성 글을 작성하기 위해 깃헙을 꾸준히 이용하고, 커밋하기 시작했습니다.) 그렇지만 글을 작성하는 습관을 길렀다 해도, 제 글이 쉽게 읽히지 않는다면 그것 또한 큰 문제겠죠. 글또는 글 작성뿐만 아니라 피드백을 통해 다른 사람들과 커뮤니케이션하는 과정이 포함되어 있습니다. 다른 분의 글을 보며 생각해볼 수 있다는 것, 제 글에 대한 피드백을 얻을 수 있다는 것은 단순히 글쓰는 습관을 기르는 것이 아닌 매력적인 글쓰기 과정이라 생각합니다. 앞으로 어떤 글을 쓰게 될지 구체적으로 정해지지는 않았지만 꾸준히 참여하고 싶습니다!



글또 4기에서 기대하시는 점은 무엇인가요? 

- 

글또 4기를 어떻게 알게 되셨나요?

- 구글에서 데이터 사이언스 관련된 글을 검색하다가, 성윤님의 블로그를 본 적이 있습니다. 정보성 글뿐만 아니라 본인의 회고록까지 다양한 글이 있어서 '이 분은 '꾸준히 블로그를 운영하시는구나. 좋은 습관이고 부러운 습관이다.'라는 생각을 하며 스크롤을 내리다가, 글또라는 모임에 대한 글을 보게 되었습니다. 이후 꼭 한 번 참여해보고 싶었습니다.

글또 4기를 6개월간 진행하며 작성할 글에 대한 계획을 최대한 자세히 작성해주세요

- 



글을 왜 작성하시려고 하시나요?

- 이성과 감성 모두를 느낀대로 잘 표현하고 싶기 때문입니다. 

글을 작성할 때 고려하는 것이 무엇인가요? 

무언가 꾸준히 노력한 경험이 있으신가요? 관련 경험에 대해 말씀해주세요

누군가에게 피드백주신 경험이 있으신가요? 글또는 리뷰 시스템을 도입해 상호리뷰를 권장하고 있습니다. 피드백을 어떻게 주셨는지, 어떻게 받았는지 등에 대해 말씀해주세요



















