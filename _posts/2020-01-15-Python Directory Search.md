---
layout: --
title:  "Python 디렉터리 검색하기"
date:   2020-01-10 14:48
categories: Python
use_math: true
---

파이썬 디렉토리 검색 방법

```python
import os

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.py': 
                    print(full_filename)
    except PermissionError:
        pass

search("c:/")
```

```python
import os

for (path, dir, files) in os.walk("c:/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.py':
            print("%s/%s" % (path, filename))
```



#확장자 추가
```python
import glob
import os.path

files = glob.glob('*')
for x in files:
    if not os.path.isdir(x):
        os.rename(x, x + '.txt')
```

#확장자 일괄 변경
```
#replace file ext.

import glob
import os.path

files = glob.glob('*.mp3')

for x in files:

    if not os.path.isdir(x):

        print x

        x2 = x.replace('.mp3', '.wav')

        print '==> ' + x2

        os.rename(x, x2)
```



#파일명 앞부분 바꾸기
```
import glob

import os.path

files = glob.glob('*.mp3')

for x in files:

    if not os.path.isdir(x):

        print x

        #print x[0]

        if x.startswith('N') == False:

            print 'not NIV'

            x2 = 'NIV-' + x

            print '==> ' + x2

            os.rename(x, x2)
```
