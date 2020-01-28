---
layout: --
title:  "PostgreSQL"
date:   2020-01-10 14:48
categories: PostgreSQL
use_math: true
---

평소에 자주 쓰는 PostgreSQL 문장을 정리해봤습니다.

### ddl문 출력하고 시
```sql
SELECT *
FROM 스키마.v_generate_tbl_ddl
WHERE tablename = LOWER('테이블명')
;
```

### Null
```
SELECT SUM(COALESCE(조건,0)) AS PCH_AMT
FROM 스키마.테이블
```


reference
- https://github.com/awslabs/amazon-redshift-utils/blob/master/src/AdminViews/v_generate_tbl_ddl.sql
- https://www.postgresqltutorial.com/postgresql-nullif/ (nullif문)
- https://augustines.tistory.com/64 (coalesce문)
