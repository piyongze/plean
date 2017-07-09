#!/usr/bin/env python3

print("Hello", "World!")

a = ["aaa", "bb"]
b = ["aaa", "bb"]
print(a is b)            ##  is 类似 java 的 ==  判断对象是否同引用
print(a == b)            ##   为值判断  类似equals

set = set()
if not set:             #### if 条件下 空序列为false
    print("abc")

s = "abc"
try:                    ##### 异常测试
    b = int(s)
except ValueError as err:
    print(err)





