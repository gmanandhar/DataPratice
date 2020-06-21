#!/usr/bin/env python

#imports

import numpy as np
import supervised.Regression.linear.Supervised_Regression_Linear_Main as lin
if __name__ == '__main__':
    lin_obj =  lin.Supervised_Regression_Linear_Main()
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])
    lin_obj.gradient_descent(x,y)




