# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:31:24 2018
`
@author: frank
"""

for i in [1,2,3]:
    print(i)
    
def muliplyby2(x):
    return x*2

def apply_to_one(f):
    return f(1)


my_list=[1,2,3,4]
my_tuple=('a','b','c','d')

test_zip=list(zip(my_list,my_tuple))

print(test_zip)


print (my_list)
print (my_tuple)

from collections import Counter
c=Counter([0,1,2,3,1,0,1])
#подсчет количества уникальных элементов в массиве
print(c)

#
#word_counts=Counter(document)



