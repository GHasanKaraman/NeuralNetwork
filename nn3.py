import numpy as np

#Multi Layer Neural Network with 2 hiden layer and output layer

X = np.array([[1,1,1],[1,0,1],[1,0,0],[0,0,1],[0,1,1],[0,1,0]]).T
Y = np.array([[1],[1],[0],[0],[1],[0]])

#a is 'or door' between first and second digit. 
#Y is 'and door' between a and the last digit.
#For example 1 0 1 = (1 or 0) and 1 = 1 and 1 = 1

p = 10   #Number of Hiden 1 Neurons
r = 10   #Number of Hiden 2 Neurons
k = 1    #Number of Output Neurons

w1 = np.random.rand(X.shape[0],p) #weight1 from inputs to hiden layer 1
H1 = 0 #Hiden layer 1

w2 = np.random.rand(p,r) #weight2 from hiden layer 1 to hiden layer 2
H2 = 0 #Hiden layer 2

w3 = np.random.rand(r,k) #weight3 from hiden layer 2 to output layer
O = 0 #output

lr = 0.1 #learning rate
epoch = 5000000

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(epoch):
    #Forward Propagation
    
    H1 = sigmoid(np.dot(X.T,w1)) #Output of Hiden layer 1
    H2 = sigmoid(np.dot(H1,w2))  #Output of Hiden layer 2
    O = sigmoid(np.dot(H2,w3))   #General Output
    
    J = (.5*(Y-O)**2).mean()*10000 #Loss Function
    
    print(i,"Loss : ",J)
    
    #Back-Propagation
    
    J_w3 = np.dot(H2.T,-(Y-O)*O*(1-O))
    J_w2 = np.dot(H1.T,H2*(1-H2)*np.dot(-(Y-O)*O*(1-O),w3.T))
    J_w1 = np.dot(X,np.dot(np.dot(-(Y-O)*O*(1-O),w3.T)*H2*(1-H2),w2.T)*H1*(1-H1))
    
    w1 = w1 - lr * J_w1
    w2 = w2 - lr * J_w2
    w3 = w3 - lr * J_w3
    
def predict(X):
    X = np.array(X)
    
    H1 = sigmoid(np.dot(X.T,w1))
    H2 = sigmoid(np.dot(H1,w2))
    O = sigmoid(np.dot(H2,w3))
    
    if O <=0.5: #threshold is 0.5
        return 0
    return 1