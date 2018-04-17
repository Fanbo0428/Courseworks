# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

#import numpy as np
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """
    grad_x=np.array([8*x[0]-2*x[1]-1,8*x[1]-2*x[0]-1])
    return grad_x
    pass

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    grad_x=np.array([(2*x[0]-2)*np.cos(x[0]*x[0]+x[1]*x[1]-2*x[0]+1)+6*x[0]-2*x[1]-2,
         (2*x[1])*np.cos(x[0]*x[0]+x[1]*x[1]-2*x[0]+1)+ 6*x[1]-2*x[0]+6])
    return grad_x
    pass

def f3(x):
     a=np.array([1.0,0.0])
     b=np.array([0.0,-1.0])
     B=np.array([[3.0,-1.0],[-1.0,3.0]])
     I=np.array([[1.0,0.0],[0.0,1.0]])
     y1=np.exp(-(np.dot(np.subtract(x,a),np.transpose(np.subtract(x,a)))))
     y2=np.exp(-(np.dot(np.dot(np.subtract(x,b),B),np.transpose(np.subtract(x,b)))))
     y3=(np.log(np.linalg.det(np.add(0.01*I,np.dot(np.transpose(x),x)))))/10.0
     #y3=np.log((x[0]*x[0]+0.01)*(x[1]*x[1]+0.01)-x[0]*x[0]*x[1]*x[1])/10.0
     return 1.0-(y1+y2-y3)
     



def f2(x):
     y = np.sin(x[0]*x[0]+x[1]*x[1]-2*x[0]+1) + 3*x[0]*x[0] + 3*x[1]*x[1] - 2*x[0]*x[1] - 2*x[0] + 6*x[1] + 3
     return y
def grad_des_f2(start_x,pace):
     #pace = 0.1
     x=start_x
     #x=np.array([[1],[-1]])
     each_x1_array=np.array([x[0]])
     each_x2_array=np.array([x[1]]) #define an array for every x used
     res_f2_array=np.array([f2(x)]) #an array for every result
     iteration = 50
     while (iteration > 1):
          grad_x=grad_f2(x)
          x=np.subtract(x,pace*grad_x)#go towards the diretion of gradient
          each_x1_array = np.append(each_x1_array,x[0])#add new x to x-array
          each_x2_array = np.append(each_x2_array,x[1])
          res_f2_array = np.append(res_f2_array,f2(x))#add new result to array
          iteration = iteration - 1
     ####################################
     #x=each_x_array[0]
     #y=each_x_array[1]
     x=np.arange(-20,20,0.01)
     y=np.arange(-20,20,0.01)
     X,Y=np.meshgrid(x,y)
     #Z=np.zeros(shape=(300,300))
     #for i in range(300):
     #     for j in range(300):
     #          Z[i][j]=f2([X[i][j],Y[i][j]])
     #print res_f2_array,each_x1_array,each_x2_array
     Z=f2([X,Y])
     ###################################
     labels = np.array(range(50))
     plt.contourf(X,Y,Z,20,alpha=0.75, cmap=plt.cm.hot)
     C=plt.contour(X,Y,Z,20,colors='black',linewidth=.5)
     plt.clabel(C,inline = True,fontsize = 10)
     plt.scatter(each_x1_array, each_x2_array,c='black', s = 10,marker='s',linewidths=1, alpha = 0.5)
     for label, x, y in zip(labels, each_x1_array, each_x2_array):
         plt.annotate(
             label,
             xy=(x, y), xytext=(-20, 20),
             textcoords='offset points', ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
     plt.xlim(-20,20)
     plt.ylim(-20,20)
     plt.show()
     #print res_f2_array
     #fig = plt.figure()
     #ax = fig.add_subplot(111, projection="3d")
     #ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
     #ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
     #ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
     #plt.show()
     
     
     '''
x=np.array([[1,2,3],[4,5,6]])
y=np.array([[1,2,3]])
tarray=np.transpose(x)
narray=np.concatenate((tarray,np.transpose(y)),axis=1)

print np.transpose(y).shape
     '''
     
grad_f3= grad(f3)

def grad_des_f3(start_x,pace):
     #pace = 0.1
     x=start_x
     #x=np.array([[1],[-1]])
     each_x1_array=np.array([x[0]])
     each_x2_array=np.array([x[1]]) #define an array for every x used
     res_f3_array=np.array([f3(x)]) #an array for every result
     iteration = 50
     while (iteration > 1):
          grad_x=grad_f3(x)
          x=np.subtract(x,pace*grad_x)#go towards the diretion of gradient
          each_x1_array = np.append(each_x1_array,x[0])#add new x to x-array
          each_x2_array = np.append(each_x2_array,x[1])
          res_f3_array = np.append(res_f3_array,f3(x))#add new result to array
          iteration = iteration - 1
     ####################################
     #x=each_x_array[0]
     #y=each_x_array[1]
     x=np.arange(-2.5,2.5,0.01)
     y=np.arange(-2.5,2.5,0.01)
     X,Y=np.meshgrid(x,y)
     Z=np.zeros(shape=(500,500))
     for i in range(500):
          for j in range(500):
               Z[i][j]=f3([X[i][j],Y[i][j]])
     #print res_f3_array,each_x1_array,each_x2_array
     #####################################
     labels = np.array(range(50))
     plt.contourf(X,Y,Z,20,alpha=0.75, cmap=plt.cm.hot)
     C=plt.contour(X,Y,Z,20,colors='black',linewidth=.5)
     plt.clabel(C,inline = True,fontsize = 10) 
     plt.scatter(each_x1_array, each_x2_array, s = 5, alpha = 0.5)

     for label, x, y in zip(labels, each_x1_array, each_x2_array):
         plt.annotate(
              label,
              xy=(x, y), xytext=(-10, 10),
              textcoords='offset points', ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
              arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
     plt.xlim(-2.5,2.5)
     plt.ylim(-2.5,2.5)
     plt.show()
     #fig = plt.figure()
     #ax = fig.add_subplot(111, projection="3d")
     #ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
     #ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
     #ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
     #plt.show()
     
##############test start##################
start_x=np.array([1.0,-1.0])
#grad_des_f2(start_x,0.3)
grad_des_f3(np.array([1.0,-1.0]),5)
##############test start##################




