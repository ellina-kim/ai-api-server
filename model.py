import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# AND 모델
class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    
     
#  OR 모델
class OrModel:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np. random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 1, 1, 1])

        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(input[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
    
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

# NOT 모델
class NotModel:
    def __init__(self):
        self.weights = np.random.rand(1)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0], [1]])
        outputs = np.array([1, 0])  # NOT 연산
        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                total_input = np.dot(inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = outputs[i] - prediction
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)
    
# XOR 모델 (PyTorch 사용)
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x
    
    def train_model(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.1)

        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(inputs)
            loss = criterion(predictions, outputs)
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

    def predict(self, input_data):
        input_tensor = torch.tensor([input_data], dtype=torch.float32)
        output = self.forward(input_tensor)
        return 1 if output.item() > 0.5 else 0