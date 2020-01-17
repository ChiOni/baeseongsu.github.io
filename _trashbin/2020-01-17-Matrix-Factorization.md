

http://www.cs.cmu.edu/~wcohen/10-601/pca+mf.pdf

https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture25-mf.pdf

https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899 // He Initialization



```python
import torch
from torch.autograd import Variable

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
	# create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
	# create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
    	# matrix multiplication
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)
```

- 여기서 n_users, n_items, n_factors는 각각 유저, 아이템, 분해요소(?)에 대한 갯를 의미
- torch.nn.Embedding() : 함수에 대한 이해가 필요해서 api를 찾아본 결과, 예시코드를 확인할 수 있었고
  - https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
  - 실행시킨 결과는 다음과 같다.
  
  ```python
       >>> # an Embedding module containing 10 tensors of size 3
       >>> embedding = nn.Embedding(10, 3)
       >>> # a batch of 2 samples of 4 indices each
       >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
       >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
    ```
  - nn.Embedding(10, 3)
    - CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None) : API 정의된 파라미터
    - num_embeddings=10, embedding_dim=3
    - num_embeddings = total number of unique elements in the vocabulary
    - embedding_dim = the size of each embedded vector once passed through the embedding layer
    - ex) We can have a tensor of 10+ elements, as long as each element in the tensor is in the range [0, 9], because we defined a vocabulary size of 10 elements.
      
  - 그러면, Embedding.weights는 어떻게 initialization 되는가?
     - initialized from $$\mathcal{N}(0, 1)N(0,1)$$
     - torch.nn.init.kaiming_normal_ 에 의해 초기화
     - gradient가 흐르게 하거나 안하게 할 수 있음 (https://discuss.pytorch.org/t/requires-grad-false-in-nn-embedding/60521)
       
  - embedding 결과값 해석
     - embedding하고 싶은 차원을 embedding_dim에 넣어주고, 단어 사전의 unique한 원소 개수를 num_embeddings에 넣어줌
     - embedding module의 row index와 해당 input value와 matching되는 embedding row vector를 가져옴 (just representation)
    
  - 다 뜯어보니 엄청 어려운 개념은 아님ㅜㅜ
  - 추가1) sparse=True인 조건에 대해서 제한된 optimizer를 사용할 수 있다고 나오는데, 여기에 해당하는 것은 현재 다음과 같음
    - optim.SGD (CUDA and CPU), optim.SparseAdam (CUDA and CPU) and optim.Adagrad (CPU)
  - 추가2) nn.Embedding.from_pretrained(weight)을 사용하면 Embedding 생성해주는 것
 
    
- 즉, 각각의 user와 item을 embedding해서 만든 embedding vector들에 대해, 곱해서주고 sum(1)을 하는 것이 forward pass를 의미
- user나 item 자체의 information을 통해 학습하는 것이 아니라 user와 item 간의 rating matrix 자체를 잘 mapping하는 user/item embedding vector를 학습하는 것


```python
model = MatrixFactorization(n_users, n_items, n_factors=20)
loss_fn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-6)

model.parameters() # parameter check

for user, item in zip(users, items):
    # get user, item and rating data
    rating = Variable(torch.FloatTensor([ratings[user, item]]))
    user = Variable(torch.LongTensor([int(user)]))
    item = Variable(torch.LongTensor([int(item)]))

    # predict
    prediction = model(user, item)
    loss = loss_fn(prediction, rating)

    # backpropagate
    loss.backward()

    # update weights
    optimizer.step()
```

- We train this model on the Movielens dataset with ratings scaled between [0, 1] to help with convergence. Applied on the test set, we obtain a root mean-squared error(RMSE) of 0.66. This means that on average, the difference between our prediction and the actual value is 0.66!
- user-item의 rating matrix의 각 element를 predictiono하여 MSELoss를 발생시킴
- SGD optimizer를 통해 2개의 embedding을 update 시킴

- 실제 데이터 응용 : Movielens dataset을 적용해봄
  - ratings : [0, 1] 사이로 scaled / minmax, standard
  - lr, epoch
  - MF (Embedding) 외에도 MF-bias 와 NNMF를 사용할 수 있음



https://jyoondev.tistory.com/42



    
    
    











