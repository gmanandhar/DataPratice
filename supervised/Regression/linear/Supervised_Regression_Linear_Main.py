#!/usr/bin/env python

"""
Linear Regression is mainly used for prediction
Example:
    Sales after 6 months

"""

#imports

import numpy as np


class Supervised_Regression_Linear_Main():
    #Default Method to initilized the value
    def __init__(self,m_curr=0, b_curr=0, iteration=1000):
        self.m_curr = m_curr
        self.b_curr = b_curr
        self.iteration = iteration
        self.learning_rate = 0.08

    def gradient_descent(self,x,y):
        n = len(x)
        for i in range(self.iteration):
            y_predicated = self.m_curr * x + self.b_curr # y=mx +b (Formula of linear regression)
            cost = (1/n) * sum([val**2 for val in (y-y_predicated)])
            md = - (2/n)* sum(x*(y-y_predicated))
            bd = - (2/n)* sum((y-y_predicated))
            self.m_curr = self.m_curr - self.learning_rate * md
            self.b_curr = self.b_curr - self.learning_rate *bd
            print("m {}, b {}, cost{}, iteration {}".format(self.m_curr,self.b_curr,cost,i))