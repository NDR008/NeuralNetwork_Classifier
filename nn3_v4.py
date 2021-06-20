import numpy as np
debug = 0  # global flag to turn on prints and plots
train = 1

class NewronLayer():
    def __init__ (self, nodes=0, inputs=0, w_data='', b_data='', output=False):
        # if output == True then SoftMax is used
        self.train_samples = None
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
        self.Z = np.dot(self.weight, self.input_data) + self.bias
        if self.final_layer:
            maxZ = np.max(self.Z)  # avoids occasional overflow warning
            # maxZ = 0
            self.A = np.exp(self.Z-maxZ) / sum(np.exp(self.Z-maxZ))  # softmax
            self.get_predictions()
        else:
            # Use ReLU as activation function
            # https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
            # https://arxiv.org/ftp/arxiv/papers/2010/2010.09458.pdf
            # Sources describe it as being the default go-to activation function            
            self.A = np.maximum(self.Z, 0)

    def backward(self, target=None):
        if self.final_layer:
            # target is the training target
            self.Softmax_backward(target)
        else:
            # target is the next layer
            self.ReLU_Backward(target)

    def get_predictions(self):
        self.predictions =  (np.argmax(self.A, 0))
    
    # this function is not actually necessary however, the code is generalised such that it can 
    # categorise for more than 2 categories.
    # this converts the target vector into an array
    # roughly like [0, 1, 2] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * might be transposed
    def encode_target(self, target):
        target = target.astype(int)
        self.encoded_target = np.zeros((target.size, int(target.max()+1)))
        length = np.arange(target.size)
        self.encoded_target[length, target] = 1
        self.encoded_target = self.encoded_target.T
        # predicted_y - target_y
        # https://peterroelants.github.io/posts/cross-entropy-softmax/
        # https://www.datahubbs.com/deep-learning-101-building-a-neural-network-from-the-ground-up/
        # Best in depth explanation: 
        # https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba

    def Softmax_backward(self, target):
        self.encode_target(target)
        self.dZ = (self.A - self.encoded_target)  # simplified to A - y, this is a 2D matrix (even if A[0] = 1-A[1])
        self.dW = 1 / self.train_samples * np.dot(self.dZ, self.input_data.T)
        self.db = 1 / self.train_samples* np.sum(self.dZ)
    
    # math form of gradient descent with back propagation:
    # https://www.datahubbs.com/deep-learning-101-building-a-neural-network-from-the-ground-up/    
    def ReLU_Backward(self, target):
        self.g_dash = self.Z > 0    # deriv func of ReLU
        self.dZ = np.dot(target.weight.T, target.dZ) * self.g_dash
        self.dW = 1 / self.train_samples * np.dot(self.dZ, self.input_data.T)
        self.db = 1 / self.train_samples * np.sum(self.dZ)

    def update(self, alpha):
        self.weight = self.weight - alpha * self.dW
        self.bias = self.bias - alpha * self.db


#layer1 = NewronLayer(4, n-1, w_data='W1.npy', b_data='b1.npy')
#layer2 = NewronLayer(2, 4, w_data='W2.npy', b_data='b2.npy', output=True)

class SpamClassifier:
    def __init__(self, NN_stucture):
        self.layers = NN_stucture
        self.layer_qty = len(self.layers)

    def train(self, max_epoch, initial_alpha, decay, train_data, test_data):
        # used to train
        train_samples, train_fields = train_data.shape
        train_data = train_data.T
        Y_train = train_data[0]
        X_train = train_data[1:train_fields]

        # use test data to judge accuracy
        test_samples, test_fields = test_data.shape
        test_data = test_data.T
        Y_test= test_data[0]
        X_test = test_data[1:test_fields]

        alpha = initial_alpha

        ### these lists are used for debugging / plotting ###
        epoch_x = []
        alpha_y = []
        ce_y2 = []
        accur_y3 = []
        ce_test_y2 = []
        accur_test_y3 = []

        ### cross entropy loss ###
        # this is only used for plotting.
        # in actual fact the cross entropy is used to calculate the loss
        # from which the back-propogation is started
        # the loss is calculated using the output of the final layer's softmax.
        # the derivative of cross_entropy(softmax(input)) is simple and does need
        # the actual cross entropy value
        # further more since the output softmax is used the cross entropy can be simplied to
        # https://www.ics.uci.edu/~pjsadows/notes.pdf
        def cross_entropy(A, Y):
            adjust = 1e-12  # avoid log(0)
            ce = -(Y * np.log(A + adjust) + (1-Y)*np.log(1 + adjust - A))
            #ce = -(Y * np.log(A))
            return np.mean(ce)

        def get_accuracy(A, Y):
            return np.sum(A == Y) / Y.size

        ########

        for epoch in range (max_epoch):
            # just for reporting
            for layer_index in range(self.layer_qty):
                if layer_index == 0:
                    self.layers[0].forward(X_train)
                else:
                    self.layers[layer_index].forward(self.layers[layer_index - 1].A)
            training_pred = self.layers[-1].predictions.copy()

            for tmp_index in range(self.layer_qty):
                layer_index = (self.layer_qty-1) - tmp_index
                self.layers[layer_index].train_samples = train_samples
                if layer_index == (self.layer_qty-1):
                    self.layers[layer_index].backward(Y_train)
                else:
                    self.layers[layer_index].backward(self.layers[layer_index+1])

            for layer_index in range(self.layer_qty):
                self.layers[layer_index].update(alpha)
            alpha = initial_alpha * (1.0 / (1 + decay * epoch))

            # test with test data
            for layer_index in range(self.layer_qty):
                if layer_index == 0:
                    self.layers[0].forward(X_test)
                else:
                    self.layers[layer_index].forward(self.layers[layer_index-1].A)
                test_pred = self.layers[-1].predictions.copy()
            
            if not debug:
                continue
            
            if epoch % 5000 == 0:
                for layer_index in range(self.layer_qty):
                    nameW = "calibration/W-" + str(layer_index) + "-" + str(epoch)
                    nameB = "calibration/B-" + str(layer_index) + "-" + str(epoch)
                    np.save(nameW, self.layers[layer_index].weight)
                    np.save(nameB, self.layers[layer_index].bias)
                print(epoch, "train", np.count_nonzero(Y_train - training_pred), "test", np.count_nonzero(Y_test - test_pred))
                            
            
            # just for reporting
            if 1:
                epoch_x.append(epoch)
                alpha_y.append(alpha)
                ce_y2.append(cross_entropy(training_pred, Y_train))
                ce_test_y2.append(cross_entropy(test_pred, Y_test))
                accur_y3.append(get_accuracy(training_pred, Y_train))
                accur_test_y3.append(get_accuracy(test_pred, Y_test))
                

        if debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2,ncols=1)
            
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_ylabel('Accuracy', color=color)
            ax1.plot(epoch_x, accur_y3, label = 'training-accuracy')
            ax1.plot(epoch_x, accur_test_y3, label = 'test-accuracy')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc="center")
                        
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Cross-Entropy Loss', color=color)
            ax2.plot(epoch_x, ce_y2, 'r', label = 'training-accuracy')
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()
            fig.savefig("calibration/entropy.png")

            print(accur_y3[-1], accur_test_y3[-1])
            
            
            

            # fig2, ax3 = plt.subplots()
            # color = 'tab:green'
            # ax3.set_xlabel('Cross-Entropy Loss')
            # ax3.set_ylabel('Accuracy', color=color)
            # ax2.tick_params(axis='y', labelcolor=color)
            # ax3.plot(ce_y2, accur_y3, color=color)
            # plt.show()
            #
            # fig3, ax4 = plt.subplots()
            # color = 'tab:purple'
            # ax4.set_xlabel('epoch (cycle)')
            # ax4.set_ylabel('Alpha', color=color)
            # ax4.plot(epoch_x, alpha_y, color=color)
            # ax4.tick_params(axis='y', labelcolor=color)
            # fig.tight_layout()
            # plt.show()


    def predict(self, data):
        data = data.T
        for layer_index in range(self.layer_qty):
            if layer_index == 0:
                self.layers[0].forward(data)
            else:
                self.layers[layer_index].forward(self.layers[layer_index-1].A)
        prediction = self.layers[-1].predictions
        return prediction


def create_classifier(receptors=54, mode=0):
    NN_stucture = []

    if mode == 0:
        NN_stucture.append(NewronLayer(4, receptors, w_data='tuned/W-0-4000.npy', b_data='tuned/B-0-4000.npy'))  
        NN_stucture.append(NewronLayer(2, 4, w_data='tuned/W-1-4000.npy', b_data='tuned/B-1-4000.npy', output=True))  

    elif mode == 1:
        layer0_nodes = 50
        layer1_nodes = 500
        layer2_nodes = 50
        NN_stucture.append(NewronLayer(layer0_nodes, receptors))  # first hidden
        NN_stucture.append(NewronLayer(layer1_nodes, layer0_nodes))  # second hidden
        NN_stucture.append(NewronLayer(layer2_nodes, layer1_nodes))  # second hidden
        NN_stucture.append(NewronLayer(2, layer2_nodes, output=True))  # output

    else:
        layer0_nodes = 15
        NN_stucture.append(NewronLayer(layer0_nodes, receptors))  
        NN_stucture.append(NewronLayer(2, layer0_nodes, output=True))  

    classifier = SpamClassifier(NN_stucture)
    return classifier

classifier = create_classifier(mode=2)  
# classifier.train(500, 0.8, 0.001, train_data, test_data)

## all for testing
if train:  
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
    test_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")
    train_data = np.array(training_spam)
    test_data = np.array(test_spam)
    X_train = train_data[:,1:]
    Y_train = train_data[:, 0]

    total = len(Y_train.T)
    correct = np.count_nonzero(Y_train - classifier.predict(X_train))
    print("at first", correct, total)
    
    classifier.train(50000, 1, 0.05, train_data, test_data)
    correct = np.count_nonzero(Y_train - classifier.predict(X_train))
    print("finally", correct, total)