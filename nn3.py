import numpy as np

training_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

train_data = np.array(training_spam)
train_data = train_data[0:-1]
m, n = train_data.shape
train_data = train_data.T
Y_train = train_data[0]
X_train = train_data[1:n]
_,m_train = X_train.shape

#print(Y_train.shape)
neurons1 = 40
neurons2 = 2  # number of classifications

def init_params():
    # randomize all around 0 +/-0.5
    # weight: row x col : to neurons x from input
    W1 = np.random.rand(neurons1, n-1) -0.5
    # bias for each hidden neural
    b1 = np.random.rand(neurons1, 1) -0.5

    # weight: row x col : to neurons x from hidden
    W2 = np.random.rand(neurons2, neurons1) -0.5
    # bias for each output neuron
    b2 = np.random.rand(neurons2, 1) -0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

# if ReLU(Z) was used, then and the output was negative,
# we lost measure of how wrong it was (or if all were negative)
def softmax(Z):
    maxZ = np.max(Z)
    # Z = Z - maxZ # to avoid exploding
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    Y = Y.astype(int)
    # if there are 3 possible values
    # vector in row0 = cases for value 0
    # vector in row1 = cases for value 1
    # vector in row2 = cases for value 2
    one_hot_Y = np.zeros((Y.size, int(Y.max()+1)))
    
    x = np.arange(Y.size)
    one_hot_Y[x, Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# even when we get an accuracy of 1, loss demonstrates how close to the ideal solution the system was
def cross_loss(soft_out, target):
    soft_out2 = np.clip(soft_out, 1e-7, 1-1e-7)  # just to remove the case of 0
    correct_conf = np.sum(soft_out2 * target, axis =0)
    loss = -(np.log(correct_conf))
    return np.mean(loss)

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):

    one_hot_Y = one_hot(Y)
    softmax_out = A2
    cl = cross_loss(softmax_out, one_hot_Y)

    # gradient descent
    dZ2 = (A2 - one_hot_Y)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1)
    a = 1
    return dW1, db1, dW2, db2, cl

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

# not hot encoded, after converting to 1D vector
def get_accuracy(predictions, Y):
    #print("P:", predictions)
    #print("Y:",Y)
    return np.sum(predictions == Y) / Y.size

def get_mismatch(predictions, Y):
    return np.count_nonzero(predictions - Y)

def gradient_descent(X, Y, initial_alpha, iterations):
    decay_rate = 0.001
    alpha = initial_alpha
    Y = Y.astype(int)
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2, cl = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 1000 == 0:
            alpha = initial_alpha * (1.0 / (1 + decay_rate * i))
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            mismatch = get_mismatch(predictions, Y)
            print("Iteration: ", i, "accuracy ", accuracy, "mismatch", mismatch, "cross-loss", cl, "learning_rate", alpha)
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1, 20000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions, A2
 
def val_prediction(index, W1, b1, W2, b2):
    #print(X_train[:, index, None])
    prediction, A2 = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print(index, "Prediction: ", prediction, "\t", "Label: ", label, "\t", "A2: ", A2[int(A2[0] > A2[1])])
    return prediction


training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
train_data = np.array(training_spam)
#train_data = train_data[300:500]
m, n = train_data.shape
train_data = train_data.T
Y_train = train_data[0]
X_train = train_data[1:n]
_,m_train = X_train.shape

counter = 0
for index in range(m):
    pred = val_prediction(index, W1, b1, W2, b2)
    if Y_train[index] != pred:
        counter += 1
print(counter, "from", m)