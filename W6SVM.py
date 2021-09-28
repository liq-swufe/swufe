import numpy as np
from sklearn.svm import  SVC
from sklearn.svm import LinearSVC
from cvxopt import matrix, solvers#二次凸规划求解
import matplotlib.pyplot as plt
import seaborn as sns
def linear_kernel(x1,x2):
    return np.dot(x1,x2)
x=np.array([[0.5,8],[1,1],[1,7],[2,2.5],[2,4],[2,6],[3,4],
            [6,2],[7,3],[7,6],[8,4],[8.5,2.5],[9,4.5]])
y=np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0])
plt.figure()
plt.scatter(x[:,0],x[:,1],c=y)

SampleNum=x.shape[0]
# 构造P矩阵
K = np.zeros((SampleNum,SampleNum))
for i in range(SampleNum):
    for j in range(SampleNum):
        K[i][j]=linear_kernel(x[i],x[j])
P=matrix(np.outer(y,y)*K)
# 构造q矩阵
q=matrix(np.ones(SampleNum) * -1) #行向量
# 构造A矩阵
A=matrix(y,(1,SampleNum))
# 构造b矩阵
b=matrix(0.0)
# 构造G矩阵
G=matrix(np.diag(np.ones(SampleNum) * -1))
# 构造h矩阵
h=matrix(np.zeros(SampleNum))
# 求解alpha
solution=solvers.qp(P,q,G,h,A,b)
alpha= np.ravel(solution['x'])
Index=alpha>1e-5 #把支持向量找出来
SupportVector=x[Index] #支持向量
alphaSV=alpha[Index] #支持向量对应的alpha找出来
ySV=y[Index]
# 求w
w=np.zeros(x.shape[1])
for i in range(len(alphaSV)):
    w=w+alphaSV[i]*ySV[i]*SupportVector[i]
# 求b
MW=np.mat(w)
MSupportVector=np.mat(SupportVector)
B=ySV-MW*MSupportVector.T
b=B.sum(axis=1)/B.shape[1]

'''
#使用python的函数包
SvcModel=SVC(C=1,kernel='linear',random_state=1)
SvcModel.fit(x,y)
print(SvcModel.support_vectors_) #支持向量
print(SvcModel.dual_coef_) # alpha 支持向量系数
print(SvcModel.coef_) # 系数
print(SvcModel.intercept_) # 截距
'''4