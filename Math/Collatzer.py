# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:05:26 2022

@author: dezyd
"""

def Collatzer(n, divider, multiplier, k = 1000):
    nList = []
    nSteps = []
    while (k>=0) and (n!=1):
        if (n % divider) == 0:
            n /= divider
            n = int(n)
            nSteps.append(n)
            nList.append("d")
        else:
            n = multiplier(n)
            nSteps.append(n)
            nList.append("m")
        k-=1
    nList = "".join(nList)
    print(nList)
    print(nSteps)
    
def StandardMultiplier(n):
    return 3*n+1