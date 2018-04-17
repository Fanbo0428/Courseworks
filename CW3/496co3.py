
# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

#import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import math
from numpy.linalg import cholesky
import random

N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
#plt.scatter(X, Y, s = 75, alpha = 0.5)

def phi1(X,order):
     result=np.zeros((N,order+1))
     for i in range(N):
          for j in range(order+1):
               result[i][j]=X[i]**j
     return result
     
def phi2_p(x,order):
     result=np.array([None]*(order+1))
     for i in range(order+1):
          if (i%2 == 0):
               result[i]=np.cos(i*math.pi*x)
          if (i%2 != 0):
               result[i]=np.sin(((i+1)/2)*math.pi*x) 
     return result
def phi2_t(X,order):
     result=np.zeros((N,order+1))
     for i in range(N):
          for j in range(order+1):
               if (j%2 == 0):
                    result[i][j]=np.cos(j*math.pi*X[i])
               if (j%2 != 0):
                    result[i][j]=np.sin(((j+1)/2)*math.pi*X[i])
     return result
     
def phi3_p(x,l,order):
     result=np.zeros((order+1,))
     m=np.linspace(-0.5,1.0,order)###order=10
     for i in range((int)(order+1)):
          if i==0:
               result[i]=1
               continue
          tmp1 = ((x-m[i-1]))**2
          tmp2 = l**2
          tmp2 = 2*tmp2
          tmp=-tmp1/tmp2
          result[i]=np.exp(tmp)
     return result
def phi3_t(X,l,order):
     result=np.zeros((N,order+1))
     m=np.linspace(-0.5,1.0,order)###order=10
     for i in range(N):
          for j in range((int)(order+1)):
               if j==0:
                    result[i][j]=1
                    continue
               tmp1 = -math.pow((X[i]-m[j-1]),2)
               tmp2 = math.pow(l,2)
               tmp2 = 2*tmp2
               tmp=tmp1/tmp2
               result[i][j]=np.exp(tmp)

     return result
     
     

def draw_lml():
     x1=np.arange(0.21,0.51,0.001)
     y1=np.arange(0.21,0.51,0.001)
     X1,Y1=np.meshgrid(x1,y1)
     Z1=np.zeros(shape=(301,301))
     for i in range(301):
          for j in range(301):
               Z1[i][j]=lml(X1[i][j],Y1[i][j],phi2_t(X,10),Y)
     plt.contourf(X1,Y1,Z1,20,alpha=0.75, cmap=plt.cm.hot)
def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """
    r1=np.linalg.det(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))**(-0.5)
    r_t=np.linalg.inv(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))
    r2=np.exp((-0.5)*np.dot(np.dot(np.transpose(Y),r_t),Y))
    r=np.log((2*np.pi)**(-Phi.shape[0]/2.0)*r1*r2)
    return r[0][0]
    pass
    
#############for gradient#####################
def lmla(alpha, beta, Phi, Y):
     r1=np.linalg.det(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))**(-0.5)
     r_t=np.linalg.inv(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))
     r2=np.exp((-0.5)*np.dot(np.dot(np.transpose(Y),r_t),Y))
     r=np.log((2*np.pi)**(-Phi.shape[0]/2.0)*r1*r2)
     return r[0][0]
def lmlb(beta,alpha,Phi,Y):
     r1=np.linalg.det(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))**(-0.5)
     r_t=np.linalg.inv(alpha*np.dot(Phi,np.transpose(Phi))+beta*np.eye(Phi.shape[0]))
     r2=np.exp((-0.5)*np.dot(np.dot(np.transpose(Y),r_t),Y))
     r=np.log((2*np.pi)**(-Phi.shape[0]/2.0)*r1*r2)
     return r[0][0]
grad_lml_alpha=grad(lmla)
grad_lml_beta=grad(lmlb)
def grad_lml(alpha, beta, Phi, Y):
     return np.array([grad_lml_alpha(alpha, beta, Phi, Y),grad_lml_beta(beta,alpha,Phi,Y)])
    
#############for gradient#####################    


def grad_des_lml(start_x,pace,iteration):
     Phi=phi1(X,1)
     x=start_x#x[0]alpha,x[1]beta
     each_alpha_array=np.zeros((iteration,))
     each_beta_array=np.zeros((iteration,)) #define an array for every (alpha,beta) used
     res_lml_array=np.zeros((iteration,)) #an array for every result
     for i in range(iteration):
          grad_x=grad_lml(x[0],x[1],Phi,Y)#find the gradient return vector
          each_alpha_array[i]=x[0]
          each_beta_array[i]=x[1]
          x=x+pace*grad_x #go towards the diretion of gradient
          res_lml_array[i]=lml(x[0],x[1],Phi,Y)
     draw_lml()
     plt.scatter(each_alpha_array,each_beta_array,s=0.1)
     plt.annotate(
         'alpha=%f,beta=%f,maxima=%f'%(each_alpha_array[iteration-1],each_beta_array[iteration-1],res_lml_array[iteration-1]),
         xy=(each_alpha_array[iteration-1],each_beta_array[iteration-1]), xytext=(-20, 20),
         textcoords='offset points', ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
     plt.show()
     #return res_lml_array,each_alpha_array,each_beta_array
     ####################################

##########grad_des for tri fun###################
def grad_des_tri_lml(start,pace,iteration,order):#np.array([0.1,0.1]),0.001,100,order
     Phi=phi2_t(X,order)
     x=start
     a=0
     b=0
     for i in range(iteration):
          grad_x=grad_lml(x[0],x[1],Phi,Y)
          x=x+pace*grad_x
          res=lml(x[0],x[1],Phi,Y)
          a=x[0]
          b=x[1]
     #return np.array([a,b])
     return res
def tri_lml_plt():
     res=np.zeros(shape=(11,))
     for i in range(11):
         res[i]=grad_des_tri_lml(np.array([0.1,0.1]),0.0001,5000,i) 
     plt.plot(np.arange(0,11,1),res,label='maximum log marginal likelihood for orders from 0 to 11')
     plt.show()
     
def gauss_lml_plt(alpha,beta,order,l):
     #order and l given by CW
     s_n=np.linalg.inv((1.0/alpha)*np.eye(order+1,order+1)+
     (1.0/beta)*np.dot(np.transpose(phi3_t(X,l,order)),phi3_t(X,l,order)))
     #m_n=np.dot(np.dot(s_n,np.transpose(phi3_t(X,l,order))),Y)
     m_n=np.dot(s_n,np.dot(np.transpose(phi3_t(X,l,order)),Y))
     m_n=10*m_n
     return m_n,s_n
     '''
     w=np.zeros((5,order+1))

     for i in range(5):
          #sn=cholesky(s_n)
          #w[i]=np.dot(np.random.randn(1,order+1), sn)+np.reshape(m_n,(11,)) 
          w[i]=np.random.multivariate_normal(np.reshape(m_n,(order+1,)),s_n)
          X_p=np.linspace(-1,1.5,200)
          Y_p=np.linspace(-1,1.5,200)

          for j in range(200):
               Y_p[j]=np.dot(w[i],np.reshape(phi3_p(X_p[j],l,order),(order+1,1)))
          #print Y_p[i]
          #Y_p[i]=np.dot(np.transpose(m_n),phi3_p(X_p[i],l,order))
          plt.plot(X_p,Y_p)
     plt.legend()
     plt.show() 
     '''

def Std(Phi_x,S_N):
    std_noise = np.sqrt(np.dot(np.dot(np.reshape(Phi_x,(1,11)),S_N), Phi_x) + 0.1)
    std_non = np.sqrt(np.dot(np.dot(np.reshape(Phi_x,(1,11)),S_N), Phi_x))
    return std_non, std_noise

def plot_prediction(m_N, S_N):
    #m_N = np.matrix(m_N)
    #S_N = np.matrix(S_N)
    plt.plot(X, Y, 'x', label = 'Training Data Points')
    test_x = np.arange(-1, 1.5, 0.001)
    test_y_m = []
    test_y_u = []
    test_y_l = []
    test_y_shade_u = []
    test_y_shade_l = []
    for x in test_x:
        #Phi_x = func_phi_GAUSSIAN(x,10)
        Phi_x= phi3_p(x,0.1,10)
        y_star_m = np.dot(np.reshape(m_N,(1,11)),np.reshape(Phi_x,(11,1)))[0][0]
        test_y_m.append(y_star_m) 
        std_non, std_noise = Std(Phi_x, S_N)
        test_y_u.append(y_star_m + 2 * std_noise)
        test_y_l.append(y_star_m - 2 * std_noise)
        test_y_shade_u.append(y_star_m + 2* std_non)
        test_y_shade_l.append(y_star_m - 2* std_non)
    #print test_y_m
    test_y_m=np.reshape(test_y_m,(2500,))
    test_y_shade_u=np.reshape(test_y_shade_u,(2500,))
    test_y_shade_l=np.reshape(test_y_shade_l,(2500,))
    plt.plot(test_x, test_y_m, color='b',label = 'Predictive Mean')

    plt.fill_between(test_x, test_y_m, test_y_shade_u, color = 'g',alpha = 0.2, label = 'Noise-Free Prediction')
    plt.plot(test_x, test_y_u, '.', label = 'Std deviation error bar with noise UPPERBOUND ')
    plt.plot(test_x, test_y_l, '.', label = 'Std deviation error bar with noise LOWERBOUND ')
    plt.fill_between(test_x, test_y_m, test_y_shade_l, color = 'g', alpha = 0.2)
    #plt.title('Predictive Mean, Error Bars and Shaded Noise-Free Prediction',fontdic)
    plt.legend()
    plt.show()

#print phi3_t(X,0.1,10).shape
#gauss_lml_plt(1,0.1,10,0.1) 
plot_prediction(gauss_lml_plt(1,0.1,10,0.1)[0],gauss_lml_plt(1,0.1,10,0.1)[1])
#tri_lml_plt()
#print grad_lml(0.5,0.5,phi1(X,1),Y)
#grad_des_lml(np.array([0.5,0.5]),0.01,200)
#grad_des_tri_lml(np.array([0.1,0.1]),0.001,100,10)
#draw_lml()
#plt.show()


