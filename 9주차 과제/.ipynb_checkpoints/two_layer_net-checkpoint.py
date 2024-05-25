class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.maxIndex = len(hidden_size)
        print("\n\n신경망 학습 시작 - 신경망: %d층"%(self.maxIndex+1))
        
        self.params = {}        
        self.params['W1'] = He_init(input_size, hidden_size[0])
        self.params['b1'] = np.zeros(hidden_size[0])

        #print(self.params['W1'].shape)

        for i in range(1, self.maxIndex):
            self.params['W' + str(i+1)] = He_init(hidden_size[i-1], hidden_size[i]) 
            self.params['b' + str(i+1)] = np.zeros(hidden_size[i])
            #print(self.params['W'  + str(i+1)].shape)

        self.params['W' + str(self.maxIndex)] = He_init(hidden_size[self.maxIndex - 1], output_size) 
        self.params['b' + str(self.maxIndex)] = np.zeros(output_size)
        #print(self.params['W' + str(self.maxIndex)].shape)

        # 계층 생성
        self.layers = OrderedDict()
        
        for i in range(1, self.maxIndex):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            self.layers['BatchNormalization' + str(i)] = BatchNormalization()
            self.layers['Relu' + str(i)] = Relu()
    
        self.layers['Affine' + str(self.maxIndex)] = Affine(self.params['W' + str(self.maxIndex)], self.params['b' + str(self.maxIndex)])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x, train_flg=True):
        for layer in self.layers.values():
            x = layer.forward(x, train_flg)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x, False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}

        for i in range(1, self.maxIndex+1):
            grads['W' + str(i)] = numerical_gradient(loss_W, self.params['W' + str(i)])
            grads['b' + str(i)] = numerical_gradient(loss_W, self.params['b' + str(i)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 계층 레이어의 값을 리스트로 변환하여 가져옴 -> 해당 리스트를 거꾸로 정렬 -> 순서대로 계층의 역전파를 실행
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 기울기
        grads = {}
        for i in range(1, self.maxIndex+1):
            grads['W' + str(i)], grads['b' + str(i)] = self.layers['Affine' + str(i)].dW, self.layers['Affine' + str(i)].db
        
        return grads