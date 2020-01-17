

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
	```python CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
	```
      - num_embeddings=10, embedding_dim=3
      - num_embeddings = total number of unique elements in the vocabulary
      - embedding_dim = the size of each embedded vector once passed through the embedding layer
      - ex) We can have a tensor of 10+ elements, as long as each element in the tensor is in the range [0, 9], because we defined a vocabulary size of 10 elements.
    - 그러면, Embedding.weights는 어떻게 initialization 되는가?
       - initialized from $$\mathcal{N}(0, 1)N(0,1)$$
       - 
    
    
    
