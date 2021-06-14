import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")

train_data = np.array(training_spam)
train_data = train_data[0:300]
m, n = train_data.shape
train_data = train_data.T
Y_train = train_data[0]
X_train = train_data[1:n]
_,m_train = X_train.shape

#print(Y_train.shape)
neurons1 = 100
neurons2 = 2

def init_params():
    W1 = np.random.rand(neurons1, n-1) *0.1
    b1 = np.random.rand(neurons1, 1) 
    W2 = np.random.rand(neurons2, neurons1) *0.1
    b2 = np.random.rand(neurons2, 1) 
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    maxZ = np.max(Z)
    Z = Z - maxZ
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    #Z1 = W1.dot(X) + b1
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    #print("1", Y)
    Y = Y.astype(int)
    #print("2", Y)
    one_hot_Y = np.zeros((Y.size, int(Y.max()+1)))
    x = np.arange(Y.size)
    one_hot_Y[x, Y] = 1
    one_hot_Y = one_hot_Y.T
    
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y) 
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print("P:", predictions)
    #print("Y:",Y)
    print(np.count_nonzero(predictions - Y))
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    Y = Y.astype(int)
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print("Iteration: ", i, "accuracy ", accuracy)
            #print("match")
            #print(predictions-Y)
            if accuracy == 1:
                break
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.2, 400)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions, A2
 
def test_prediction(index, W1, b1, W2, b2):
    #print(X_train[:, index, None])
    prediction, A2 = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print(index, "Prediction: ", prediction, "\t", "Label: ", label, "\t", "A2: ", A2[0], A2[1])
    return prediction

train_data = np.array(training_spam)
train_data = train_data[300:500]
m, n = train_data.shape
train_data = train_data.T
Y_train = train_data[0]
X_train = train_data[1:n]
_,m_train = X_train.shape

counter = 0
for index in range(m):
    pred = test_prediction(index, W1, b1, W2, b2)
    if Y_train[index] != pred:
        counter += 1
print(counter)


class NewronLayer:
    def __init__(self, receptors, newrons) :
        self.weight = np.random.rand(neurons1, n-1) - 0.5
        self.bias = np.random.rand(neurons1, 1) - 0.5
