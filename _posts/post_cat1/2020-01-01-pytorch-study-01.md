---
layout: post
title:  "Pytorch Study 01"
date:   2020-01-01 17:27
categories: Pytorch
---
Chapter 1 Introducing deep learning and the PyTroch library

Chapter 2 It starts with a tensor

Chapter 3 Real-world data representation with tensors

Chapter 4 The mechanics of learning

Chapter 5 Using a nueral network to fit your data



## Chapter 2 *It starts with a tensor*

- 텐서, 파이토치의 기본 자료 구조
- Indexing and operating on PyTorch tensors to explore and manpulate data
- Interoperating with NumPy multidimensional arrays
- 속도를 위해 연산을 GPU로 이동

<소개글>



### *2.1 Tensors fundamentals*

A tensor is an array—that is, a data structure storing collection of numbers that are accessible individually by means of an index and that can be indexed with multiple indices.

파이썬 list indexing

- Numbers in Python are full-fledged objects?
- Lists in Python are meant for sequential collections of objects?
- The Python interpreter is slow compared with optimzed, compiled code?
- For these reasons, data science libraries rely on NumPy or introduce dedicated data structures such as PyTorch tensors 

```python

# In[1]:
a = [1.0, 2.0, 1.0]

# In[2]:
a[0]

# Out[2]:
1.0

# In[3]:
a[2] = 3.0
a

# Out[3]:
[1.0, 2.0, 3.0]

```



파이토치 tensor indexing

```python

# In[4]:
import torch
a = torch.ones(3)
a

# Out[4]:
tensor([1., 1., 1.])

# In[5]:
a[1]

# Out[5]:
tensor(1.)

# In[6]:
float(a[1])

# Out[6]:
1.0

# In[7]:
a[2] = 2.0
a

# Out[7]:
tensor([1., 1., 2.])


```



겉으로는 파이썬 리스트와 파이토치 텐서와 다르지 않아보이지만, 실제로 뜯어보면 완전히 다름. 

<img src="/Users/skcc10170/Library/Application Support/typora-user-images/image-20200101175315313.png" alt="image-20200101175315313" style="zoom:33%;" />

- 파이썬 리스트나 숫자들의 튜플 형태는 메모리에 개별적으로 할당된 파이썬 객체들의 집합 (boxed)
- 파이토치 텐서나 넘파이 어레이 형태는 contiguous memory blocks containing unboxed C numeric types, not Python objects임. In this case, 32 bits (4 bytes) float, as you see on the right side of figure 2.3. So a 1D tensor of 1 million float numbers requires 4 million contiguous bytes to be stored, plus a small overhead for the metadata (dimensions, numeric type, and so on).



coordinate - 1d tensor

```python

points = torch.zeros(6)
points[0] = 1.0
points[0] = 1.0
points[0] = 1.0
points[0] = 1.0
points[0] = 1.0
points[0] = 1.0

```



```python

points = torch.tensor([1.0, 4.0, 2.0, 1.0, 3.0, 5.0])

```



````
float(points[0]), float(points[1])

(1.0, 4.0)
````



upgrade ver. using 2d tensor

- by passing a list of lists to the constructor

```python

points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points

tensor([[1., 4.],
        [2., 1.],
        [3., 5.]])

---

points.shape

torch.Size([3, 2])

---

points = torch.zeros(3, 2)
points

tensor([0., 0.],
       [0., 0.],
       [0., 0.])

---

points = torch.FloatTensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
points

tensor([[1., 4.],
        [2., 1.],
        [3., 5.]])

points[0, 1] # returns the y coordinate of the 0th point in your data set
# 0-based index, 처음자리는 데이터 갯수, 두번째는 x/y값인지 구분

tensor(4.)


---

points[0]

tensor([1., 4.])


```



Note that what you get as the output is *another tensor*, but a 1D tensor of size 2 contain- ing the values in the first row of the points tensor. Does this output mean that a new chunk of memory was allocated, values were copied into it, and the new memory was returned wrapped in a new tensor object? No, because that process would be ineffi- cient, especially if you had millions of points. What you got back instead was a differ- ent *view* of the same underlying data, limited to the first row.







### *2.2 Tensors and storages*

storages라는 객체가 있는데 tensor는 이러한 형태로 저장됨



### 2.3 *Size, storage offset, and strides*







Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse

```javascript
const Razorpay = require('razorpay');

let rzp = Razorpay({
	key_id: 'KEY_ID',
	secret: 'name'
});

// capture request
rzp.capture(payment_id, cost)
	.then(function (data) {
		return 2;
	})
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/