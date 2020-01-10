---
layout: post
title:  "PostgreSQL"
date:   2020-01-10 14:48
categories: PostgreSQL
use_math: true
---


```sql
SELECT *
FROM 스키마.v_generate_tbl_ddl
WHERE tablename = LOWER('테이블명')
;
```
