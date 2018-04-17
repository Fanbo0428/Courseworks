import numpy as np
from scipy.linalg import solve 
from scipy import stats
import matplotlib.pyplot as plt  
import math   
import random
from sklearn.model_selection import train_test_split 


N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
##annoted when cv
plt.scatter(X, Y, s = 75, alpha = 0.5)
plt.xlim(-0.3,1.3)
plt.ylim(-1.5,2)

###the poly funtion phi1
def phi1(x,order):
     result=np.array([None]*(order+1))
     for i in range(order+1):
          result[i]=math.pow(x,i)
     return result
###the tri funtion phi2
def phi2(x,order):
     result=np.array([None]*(order+1))
     for i in range(order+1):
          if (i%2 == 0):
               result[i]=np.cos(i*math.pi*x)
          if (i%2 != 0):
               result[i]=np.sin(((i+1)/2)*math.pi*x)
               #print result[i] 
     return result
     
###the Gauss funtion phi3
def phi3(x,l,order):
     result=np.array([None]*(int)(order+1))
     m=np.linspace(0,1,order+1)###order=19
     for i in range((int)(order+1)):
          if i==0:
               result[i]=1
          tmp1 = -math.pow((x-m[i]),2)
          tmp2 = math.pow(l,2)
          tmp2 = 2*tmp2
          tmp=tmp1/tmp2
          result[i]=np.exp(tmp)
     return result


##############################################
def polynomial(polyorder,s):
     w_matrix=np.zeros(shape=(polyorder+1,polyorder+1))
     for i in range(polyorder+1):
          for x in X:
               w_matrix[i] = w_matrix[i] + phi1(x,polyorder)*phi1(x,polyorder)[i]
     w_r_matrix=np.zeros(shape=(polyorder+1,))
     for i in range(polyorder+1):
          for j in range(25):
               w_r_matrix[i]=w_r_matrix[i]+Y[j]*phi1(X[j],polyorder)[i]
     #print w_matrix,w_r_matrix
     w=solve(w_matrix,w_r_matrix)
     #print w
     d=0
     for i in range(25):
          d=d+math.pow(Y[i]-np.dot(np.transpose(w),np.reshape(phi1(X[i],polyorder),(polyorder+1,1))),2)
     d=d/25
     xt=np.linspace(-0.3, 1.3, 200)
     yt=np.linspace(1, 2, 200)
     for i in range(200):
          #yt[i]=random.gauss(np.dot(np.transpose(w),np.reshape(phi1(xt[i],polyorder),(polyorder+1,1)))[0],d)
          yt[i]=np.dot(np.transpose(w),np.reshape(phi1(xt[i],polyorder),(polyorder+1,1)))[0]
     plt.plot(xt,yt,label=s)
##############################################
def polytr(polyorder,X,Y):
     w_matrix=np.zeros(shape=(polyorder+1,polyorder+1))
     for i in range(polyorder+1):
          for x in X:
               w_matrix[i] = w_matrix[i] + phi2(x,polyorder)*phi2(x,polyorder)[i]
     w_r_matrix=np.zeros(shape=(polyorder+1,))
     for i in range(polyorder+1):
          for j in range(len(X)):
               w_r_matrix[i]=w_r_matrix[i]+Y[j]*phi2(X[j],polyorder)[i]
     w=solve(w_matrix,w_r_matrix)
     d=0
     for i in range(len(X)):
          d=d+math.pow(Y[i]-np.dot(np.transpose(w),np.reshape(phi2(X[i],polyorder),(polyorder+1,1))),2)
     d=d/25
     #annoted when cv
     xt=np.linspace(-0.3, 1.3, 200)
     yt=np.linspace(1, 2, 200)
     for i in range(200):
          yt[i]=random.gauss(np.dot(np.transpose(w),np.reshape(phi2(xt[i],polyorder),(polyorder+1,1)))[0],d)
     plt.plot(xt,yt)
     return w,d

def ridge(gaussorder,lamda,l,label_s):
     w_matrix=np.zeros(shape=(gaussorder+1,gaussorder+1))
     for i in range((int)(gaussorder+1)):
          for x in X:
               w_matrix[i] = w_matrix[i] + phi3(x,l,gaussorder)*phi3(x,l,gaussorder)[i]
     w_matrix=w_matrix+lamda*np.eye(gaussorder+1)
     w_r_matrix=np.zeros(shape=(gaussorder+1,))
     for i in range(gaussorder+1):
          for j in range(25):
               w_r_matrix[i]=w_r_matrix[i]+Y[j]*phi3(X[j],l,gaussorder)[i]
     
     w=solve(w_matrix,w_r_matrix)
     xt=np.linspace(-0.3, 1.3, 200)
     yt=np.linspace(1, 2, 200)
     for i in range(200):
          yt[i]=np.dot(np.transpose(w),np.reshape(phi3(xt[i],l,gaussorder),(gaussorder+1,1)))
     plt.plot(xt,yt,label=label_s)


def cvintr(cvorder):
     err=0
     d=0
     for i in range(N):
          x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.04,random_state=1)
          w,d = polytr(cvorder,x_train,y_train)[0],polytr(cvorder,x_train,y_train)[1]##this funtion returns 2 values, w and d
          y_t=random.gauss(np.dot(np.transpose(w),np.reshape(phi2(x_test,cvorder),(cvorder+1,1)))[0],d)
          err=err+(np.cos(10*x_test**2) + 0.1 * np.sin(100*x_test)-y_t)**2
     err=err/25.0
     return err,d    

def cvplt():
     err=np.array([None]*11)
     da=np.array([None]*11)
     for i in range(11):
          err[i]=cvintr(i)[0]
          da[i]=cvintr(i)[1]
     xp=np.linspace(0,10,11)
     yp=np.linspace(0,10,11)
     yd=np.linspace(0,10,11)
     for i in range(11):
          yp[i]=err[i]
          yd[i]=da[i]
     plt.plot(xp,yp,label="$square erro$")
     plt.plot(xp,yd,label="$variance$")
     plt.xlim(0.0,10)
     plt.ylim(0,0.2)
     pass




#ridge(20,0.000,0.1,"over-fitting with lamda=0.0")
#ridge(20,10,0.1,"under fitting with lamda=10")
#ridge(20,0.1,0.1,"reasonable fitting with lamda=0.1")
#cvplt()
polynomial(0,"order=0")
polynomial(1,"order=1")
polynomial(2,"order=2")
polynomial(3,"order=3")
polynomial(11,"order=11")
#polytr(1,X,Y,"order=1")
#polytr(11,X,Y,"order=11")
plt.legend()
plt.show()
 
     
#Z=np.zeros(shape=(500,500))

