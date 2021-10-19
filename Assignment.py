#!/usr/bin/env python
# coding: utf-8

# # <font color='red'>Implement SGD Classifier with Logloss and L2 regularization Using SGD without using sklearn</font>

# **There will be some functions that start with the word "grader" ex: grader_weights(), grader_sigmoid(), grader_logloss() etc, you should not change those function definition.<br><br>Every Grader function has to return True.**

# <font color='red'> Importing packages</font>

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


# <font color='red'>Creating custom dataset</font>

# In[2]:


# please don't change random_state
X, y = make_classification(n_samples=50000, n_features=15, n_informative=10, n_redundant=5,
                           n_classes=2, weights=[0.7], class_sep=0.7, random_state=15)
# make_classification is used to create custom dataset 
# Please check this link (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) for more details


# In[3]:


X.shape, y.shape


# <font color='red'>Splitting data into train and test </font>

# In[4]:


#please don't change random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)


# In[5]:


# Standardizing the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 


# In[6]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # <font color='red' size=5>SGD classifier</font>

# In[7]:


# alpha : float
# Constant that multiplies the regularization term. 

# eta0 : double
# The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules.

clf = linear_model.SGDClassifier(eta0=0.0001, alpha=0.0001, loss='log', random_state=15, penalty='l2', tol=1e-3, verbose=2, learning_rate='constant')
clf


# Please check this documentation (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
#eta0 defines learning rate taken to be constant
#penality gives Regularization if penality is 'l2' then it is l2 regularization
#loss helps to create linear models giving it to SGD classifier. if loss='log' it is logistic loss
#alpha constant that multiplies with regularizer to make it stronger
#tol is the stopping criterion If it is not None, training will stop when (loss > best_loss - tol) 


# In[8]:


clf.fit(X=X_train, y=y_train) # fitting our model


# In[9]:


clf.coef_, clf.coef_.shape, clf.intercept_
#clf.coef_ will return the weights
#clf.coef_.shape will return the shape of weights
#clf.intercept_ will return the intercept term


# 
# 
# ```
# # This is formatted as code
# ```
# 
# ## <font color='red' size=5> Implement Logistic Regression with L2 regularization Using SGD: without using sklearn </font>
# 
# 

# 
# 
# 
# 1.  We will be giving you some functions, please write code in that functions only.
# 
# 2.  After every function, we will be giving you expected output, please make sure that you get that output. 
# 
# 
# 
# 

# 
# <br>
# 
# * Initialize the weight_vector and intercept term to zeros (Write your code in <font color='blue'>def initialize_weights()</font>)
# 
# * Create a loss function (Write your code in <font color='blue'>def logloss()</font>) 
# 
#  $log loss = -1*\frac{1}{n}\Sigma_{for each Yt,Y_{pred}}(Ytlog10(Y_{pred})+(1-Yt)log10(1-Y_{pred}))$
# - for each epoch:
# 
#     - for each batch of data points in train: (keep batch size=1)
# 
#         - calculate the gradient of loss function w.r.t each weight in weight vector (write your code in <font color='blue'>def gradient_dw()</font>)
# 
#         $dw^{(t)} = x_n(y_n − σ((w^{(t)})^{T} x_n+b^{t}))- \frac{λ}{N}w^{(t)})$ <br>
# 
#         - Calculate the gradient of the intercept (write your code in <font color='blue'> def gradient_db()</font>) <a href='https://drive.google.com/file/d/1nQ08-XY4zvOLzRX-lGf8EYB5arb7-m1H/view?usp=sharing'>check this</a>
# 
#            $ db^{(t)} = y_n- σ((w^{(t)})^{T} x_n+b^{t}))$
# 
#         - Update weights and intercept (check the equation number 32 in the above mentioned <a href='https://drive.google.com/file/d/1nQ08-XY4zvOLzRX-lGf8EYB5arb7-m1H/view?usp=sharing'>pdf</a>): <br>
#         $w^{(t+1)}← w^{(t)}+α(dw^{(t)}) $<br>
# 
#         $b^{(t+1)}←b^{(t)}+α(db^{(t)}) $
#     - calculate the log loss for train and test with the updated weights (you can check the python assignment 10th question)
#     - And if you wish, you can compare the previous loss and the current loss, if it is not updating, then
#         you can stop the training
#     - append this loss in the list ( this will be used to see how loss is changing for each epoch after the training is over )
# 

# In[10]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


# In[11]:


#as already we have taken custom dataset lets take that
#we use make_classification() to crete custom dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
X,y=make_classification(n_samples=50000,n_features=15,n_informative=10,n_redundant=5,n_classes=2,weights=[0.7],class_sep=0.7,random_state=15)
#splitting Train and Test dataset
#we use train_test_split() to split train and test dataset https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=15)


# <font color='blue'>Initialize weights </font>

# In[12]:


def initialize_weights(dim):
    ''' In this function, we will initialize our weights and bias'''
    #initialize the weights to zeros array of (1,dim) dimensions
    #you use zeros_like function to initialize zero, check this link https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros_like.html
    #initialize bias to zero
    #we are going to initialize both objective function weighted vector and intercept
    w=np.zeros_like(X_train[0])
    b=0

    return w,b


# In[13]:


dim=X_train[0] 
w,b = initialize_weights(dim)
print('w =',(w))
print('b =',str(b))


# <font color='cyan'>Grader function - 1 </font>

# In[14]:


dim=X_train[0] 
w,b = initialize_weights(dim)
def grader_weights(w,b):
  assert((len(w)==len(dim)) and b==0 and np.sum(w)==0.0)
  return True
grader_weights(w,b)


# <font color='blue'>Compute sigmoid </font>

# $sigmoid(z)= 1/(1+exp(-z))$

# In[15]:


#Here to generate binary values 0 or 1 we use sigmoid function
def sigmoid(z):
    ''' In this function, we will return sigmoid of z'''
    # compute sigmoid(z) and return

    return 1/(1+np.exp(-z))


# <font color='cyan'>Grader function - 2</font>

# In[16]:


def grader_sigmoid(z):
  val=sigmoid(z)
  assert(val==0.8807970779778823)
  return True
grader_sigmoid(2)


# <font color='blue'> Compute loss </font>

# $log loss = -1*\frac{1}{n}\Sigma_{for each Yt,Y_{pred}}(Ytlog10(Y_{pred})+(1-Yt)log10(1-Y_{pred}))$

# In[17]:


#https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/
def logloss(y_true,y_pred):
    '''In this function, we will compute log loss '''
    #initializing the sum
    sum = 0
    for i in range(len(y_true)):
        sum+=(y_true[i]*np.log10(y_pred[i])) + ((1-y_true[i]) * np.log10(1-y_pred[i]))
    loss = -1 * (1/len(y_true)) * sum

    return loss


# In[ ]:





# <font color='cyan'>Grader function - 3 </font>

# In[18]:


def grader_logloss(true,pred):
  loss=logloss(true,pred)
  assert(loss==0.07644900402910389)
  return True
true=[1,1,0,1,0]
pred=[0.9,0.8,0.1,0.8,0.2]
grader_logloss(true,pred)


# <font color='blue'>Compute gradient w.r.to  'w' </font>

# $dw^{(t)} = x_n(y_n − σ((w^{(t)})^{T} x_n+b^{t}))- \frac{λ}{N}w^{(t)}$ <br>

# In[19]:


def gradient_dw(x,y,w,b,alpha,N):
    '''In this function, we will compute the gardient w.r.to w '''
    dw=x * (y-sigmoid(np.dot(w,x) + b)-(alpha/N)*w)

    return dw


# <font color='cyan'>Grader function - 4 </font>

# In[20]:


def grader_dw(x,y,w,b,alpha,N):
  grad_dw=gradient_dw(x,y,w,b,alpha,N)
  assert(np.sum(grad_dw)==2.613689585)
  return True
grad_x=np.array([-2.07864835,  3.31604252, -0.79104357, -3.87045546, -1.14783286,
       -2.81434437, -0.86771071, -0.04073287,  0.84827878,  1.99451725,
        3.67152472,  0.01451875,  2.01062888,  0.07373904, -5.54586092])
grad_y=0
grad_w,grad_b=initialize_weights(grad_x)
alpha=0.0001
N=len(X_train)
grader_dw(grad_x,grad_y,grad_w,grad_b,alpha,N)


# <font color='blue'>Compute gradient w.r.to 'b' </font>

# $ db^{(t)} = y_n- σ((w^{(t)})^{T} x_n+b^{t})$

# In[21]:


def gradient_db(x,y,w,b):
    '''In this function, we will compute gradient w.r.to b '''
    db=y-sigmoid(np.dot(w,x)+b)

    return db


# <font color='cyan'>Grader function - 5 </font>

# In[22]:


def grader_db(x,y,w,b):
  grad_db=gradient_db(x,y,w,b)
  assert(grad_db==-0.5)
  return True
grad_x=np.array([-2.07864835,  3.31604252, -0.79104357, -3.87045546, -1.14783286,
       -2.81434437, -0.86771071, -0.04073287,  0.84827878,  1.99451725,
        3.67152472,  0.01451875,  2.01062888,  0.07373904, -5.54586092])
grad_y=0
grad_w,grad_b=initialize_weights(grad_x)
alpha=0.0001
N=len(X_train)
grader_db(grad_x,grad_y,grad_w,grad_b)


# <font color='blue'> Implementing logistic regression</font>

# In[28]:


def train(X_train,y_train,X_test,y_test,epochs,alpha,eta0):
    ''' In this function, we will implement logistic regression'''
    #initialize train_loss and test_loss 
    train_loss=[]
    test_loss=[]
    #initialize weights and intercept
    w,b=initialize_weights(X_train[0])
    for i in range(epochs):
        train_pred=[]
        test_pred=[]
        for j in range(N):
            dw=gradient_dw(X_train[j],y_train[j],w,b,alpha,N)
            db=gradient_db(X_train[j],y_train[j],w,b)
            w=w+(eta0 * dw)
            b=b+(eta0 * db)
        for val in range(N):
            train_pred.append(sigmoid(np.dot(w,X_train[val])+b))
            
        loss1=logloss(y_train, train_pred)
        train_loss.append(loss1)
        
        for val in range(len(X_test)):
            test_pred.append(sigmoid(np.dot(w,X_test[val])+b))
            
        loss2=logloss(y_test,test_pred)
        test_loss.append(loss2)
        
    return w,b,train_loss,test_loss
    #Here eta0 is learning rate
    #implement the code as follows
    # initalize the weights (call the initialize_weights(X_train[0]) function)
    # for every epoch
        # for every data point(X_train,y_train)
           #compute gradient w.r.to w (call the gradient_dw() function)
           #compute gradient w.r.to b (call the gradient_db() function)
           #update w, b
        # predict the output of x_train[for all data points in X_train] using w,b
        #compute the loss between predicted and actual values (call the loss function)
        # store all the train loss values in a list
        # predict the output of x_test[for all data points in X_test] using w,b
        #compute the loss between predicted and actual values (call the loss function)
        # store all the test loss values in a list
        # you can also compare previous loss and current loss, if loss is not updating then stop the process and return w,b


# In[29]:


alpha=0.0001
eta0=0.0001
N=len(X_train)
epochs=50
w,b,train_loss,test_loss=train(X_train,y_train,X_test,y_test,epochs,alpha,eta0)


# <font color='red'>Goal of assignment</font>

# Compare your implementation and SGDClassifier's the weights and intercept, make sure they are as close as possible i.e difference should be in terms of 10^-3

# In[30]:


# these are the results we got after we implemented sgd and found the optimal weights and intercept
w-clf.coef_, b-clf.intercept_


# <font color='blue'>Plot epoch number vs train , test loss </font>
# 
# * epoch number on X-axis
# * loss on Y-axis

# In[ ]:





# In[31]:


def pred(w,b, X):
    N = len(X)
    predict = []
    for i in range(N):
        z=np.dot(w,X[i])+b
        if sigmoid(z) >= 0.5: # sigmoid(w,x,b) returns 1/(1+exp(-(dot(x,w)+b)))
            predict.append(1)
        else:
            predict.append(0)
    return np.array(predict)
print(1-np.sum(y_train - pred(w,b,X_train))/len(X_train))
print(1-np.sum(y_test  - pred(w,b,X_test))/len(X_test))


# In[ ]:





# In[ ]:




