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
- torch.nn.Embedding() : 함수에 대한 이해가 필요해서 api를 찾아본 결과
	- 
