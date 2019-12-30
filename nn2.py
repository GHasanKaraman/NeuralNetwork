import numpy as np

#Multi Layer Neural Netweork with 1 hiden layer and output layer

X = np.array([[1,0,0],[1,1,0],[1,1,1],[0,0,0],[0,0,1],[0,1,1]]).T
Y = np.array([[1],[1],[1],[0],[0],[0]])

#The answer is always first digit
#For example 1,1,1 = 1   0,1,0 = 0

p = 3  #Number of Hiden Neurons
k = 1   #Number of Output Neurons

w_ = np.random.rand(X.shape[0],p) #weight1 from inputs to hiden layer
H = 0 #Hiden layer
_w_ = np.random.rand(p,k) #weight2 from hiden layer to output layer

lr = 0.01 #Learning rate
epoch = 100000

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(epoch):
    #Forward Propagation
    
    H = np.dot(X.T,w_)
    Out_H = sigmoid(H)
    O = np.dot(Out_H,_w_)
    Out_O = sigmoid(O)
    
    J = (.5*(Y-Out_O)**2).mean()*100 #Loss Function
    
    print(i,"Loss : ",J)
    
    #Back-Propagation
    
    J_w__ = np.dot(H.T,-(Y-O)*Out_O*(1-Out_O))
    
    J_w_ = np.dot(X,np.dot(-(Y-O)*Out_O*(1-Out_O),_w_.T)*Out_H*(1-Out_H))
    
    _w_ = _w_ - lr * J_w__
    
    w_ = w_ - lr * J_w_
    
def predict(x):
    x = np.array(x)
    
    out_h = sigmoid(np.dot(x.T,w_))
    out_o = sigmoid(np.dot(out_h,_w_))
    
    if out_o <=0.65: #threshold is .65
        return 0
    return 1