import numpy as np

class NewronLayer():
    def __init__ (self, nodes=0, inputs=0, w_data='', b_data='', output=False):
        # if output == True then SoftMax is used
        self.inputs = int(inputs)
        self.nodes = int(nodes)
        self.final_layer = output
        # # weight: row x col : to neurons x from input
        # # weight: row x col : to neurons x from hidden
        if len(w_data) == 0:
            self.weight = np.random.randn(self.nodes, self.inputs) * 0.1
            self.bias = np.random.randn(self.nodes, 1) * 0.1
        else:
            self.weight = np.load(w_data).copy()
            self.bias = np.load(b_data).copy()
    
    def forward(self, data):
        self.input_data = data.copy()  # only useful for training or debug
        self.Z = np.dot(self.weight, data) + self.bias
        if self.final_layer:
            self.A = np.exp(self.Z) / sum(np.exp(self.Z))
            self.get_predictions()
        else:            
            self.A = np.maximum(self.Z, 0)

    def backward(self, target=None):
        if self.final_layer:
            # target is the training target
            self.Softmax_backward(target)
        else:
            # target is the next layer
            self.ReLU_Backward(target)

    def get_predictions(self):
        self.preductions =  np.argmax(self.A, 0)
    
    def encode_target(self, target):
        target = target.astype(int)
        self.encoded_target = np.zeros((target.size, int(target.max()+1)))
        length = np.arange(target.size)
        self.encoded_target[length, target] = 1
        self.encoded_target = self.encoded_target.T
        # It is said that the derivative of the cross-entrop loss function for the softmax function is
        # predicted_y - target_y
        # https://peterroelants.github.io/posts/cross-entropy-softmax/

    def Softmax_backward(self, target):
        self.encode_target(target)
        self.dZ = (self.A - self.encoded_target)
        self.dW = 1 / m * np.dot(self.dZ, self.input_data.T)
        self.db = 1 / m * np.sum(self.dZ)


    def ReLU_Backward(self, target):
        self.dA = self.Z > 0
        self.dZ = np.dot(target.weight.T, target.dZ) * self.dA
        self.dW = 1 / m * np.dot(self.dZ, self.input_data.T)
        self.db = 1 / m * np.sum(self.dZ)

    def update(self):
        self.weight = self.weight - alpha * self.dW
        self.bias = self.bias - alpha * self.db
        

        
training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
train_data = np.array(training_spam)
m, n = train_data.shape
train_data = train_data.T
Y_train = train_data[0]
X_train = train_data[1:n]

#layer1 = NewronLayer(4, n-1, w_data='W1.npy', b_data='b1.npy')
#layer2 = NewronLayer(2, 4, w_data='W2.npy', b_data='b2.npy', output=True)

print("start", np.count_nonzero(layer2.preductions != Y_train))
for i in range(1000):
    layer1.forward(X_train)
    layer2.forward(layer1.A)
    print(i, np.count_nonzero(layer2.preductions != Y_train))

    layer2.backward(target=Y_train)
    layer1.backward(layer2)

    alpha = 0.1
    layer1.update()
    layer2.update()

layer1.forward(X_train)
layer2.forward(layer1.A)

class SpamClassifier:
    def __init__(self, NN_stucture):
        self.layers = NN_stucture
    def train(self):
        pass
    def predict(self, data):
        self.layers[0].forward(data)
        self.layers[1].forward(self.layers[0].A)

def create_classifier():
    layer1_nodes = 50
    NN_stucture = {}
    NN_stucture[0] = NewronLayer(nodes=layer1_nodes, inputs=n - 1)  # 4 neurons for the hidden layer with n receptors
    NN_stucture[1] = NewronLayer(nodes=2, inputs=layer1_nodes, output=True)  # 2 neurons for the ouput layer with 4 receptors

    classifier = SpamClassifier(NN_stucture)
    #classifier.train()
    return classifier

classifier = create_classifier()

