import numpy as np

#Perceptron

x = np.array([[1,0,0],[1,1,0],[1,1,1],[0,0,0],[0,0,1],[0,1,1]]).T
y = np.array([[1],[1],[1],[0],[0],[0]])

#The answer is always first digit 
#For example 1,1,1 = 1   0,1,0 = 0

w = np.zeros((x.shape[0],1)) #weight

b = 0 #bias

lr = 0.01 #learning rate
epoch = 10000 

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(epoch):
    #Forward Propagation
    
    output = sigmoid(np.dot(x.T,w)+b)
    
    J = (.5*(y-output)**2).mean() #Loss function
    
    print(i,"Loss : ",J)
    
    #Back-Propagation
    
    J_Output = -(y-output)
    Output_O = output*(1-output)
    
    J_w = np.dot(x,J_Output*Output_O)
    J_b = J_Output*Output_O
    
    w = w - lr * J_w
    b = b - lr * J_b
    
def predict(data):
    data = np.array(data)
    
    x = sigmoid(np.dot(data,w)+b.mean())
    
    if(x <= 0.5): #threshold is 0.5
        return 0
    return 1