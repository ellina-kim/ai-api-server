import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

# 모델 저장 형식 선택
MODEL_SAVE_FORMAT = "json"  # AND, OR, NOT 모델 저장용 (JSON 방식)

# ✅ AND, OR, NOT 모델 (NumPy 기반)
class LogicModel:
    def __init__(self, model_name, inputs, outputs):
        self.model_name = model_name
        self.weights = np.random.rand(inputs.shape[1])  # 가중치 초기화
        self.bias = np.random.rand(1)
        self.inputs = inputs
        self.outputs = outputs

    def train(self):
        learning_rate = 0.1
        epochs = 20
        for epoch in range(epochs):
            for i in range(len(self.inputs)):
                total_input = np.dot(self.inputs[i], self.weights) + self.bias
                prediction = self.step_function(total_input)
                error = self.outputs[i] - prediction
                self.weights += learning_rate * error * self.inputs[i]
                self.bias += learning_rate * error
        self.save_model()

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)

    def save_model(self):
        model_data = {
            "weights": self.weights.tolist(), # 같은 레벨(위치)에 있는 데이터끼리 묶어줌. 
            "bias": self.bias.tolist() # NumPy 배열 -> 리스트로 변환 후 저장
        }
        with open(f"{self.model_name}.json", "w") as f:
            json.dump(model_data, f)
        print(f"{self.model_name} model saved as JSON.")

    def load_model(self):
        if os.path.exists(f"{self.model_name}.json"):
            with open(f"{self.model_name}.json", "r") as f:
                model_data = json.load(f)
                self.weights = np.array(model_data["weights"])
                self.bias = np.array(model_data["bias"])
            print(f"{self.model_name} model loaded from JSON.")
        else:
            print(f"No pre-trained {self.model_name} model found.")

# ✅ PyTorch 기반 XOR 모델
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

    def train_model(self):
        inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.1)

        epochs = 2000
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(inputs)
            loss = criterion(predictions, outputs)
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

        torch.save(self.state_dict(), "xor_model.pth")
        print("XOR model saved as xor_model.pth.")

    def load_model(self):
        if os.path.exists("xor_model.pth"):
            try:
                self.load_state_dict(torch.load("xor_model.pth"))  # ✅ 기존 모델 불러오기
                self.eval()
                print("XOR model loaded successfully from xor_model.pth.")
            except RuntimeError:  # 🚨 모델 크기가 다르면 다시 학습!
                print("Model structure mismatch! Re-training the model...")
                self.train_model()  # ✅ 다시 학습 후 저장
        else:
            print("No pre-trained XOR model found. Training a new model...")
            self.train_model()

    def predict(self, input_data):
        input_tensor = torch.tensor([input_data], dtype=torch.float32)
        output = self.forward(input_tensor)
        return 1 if output.item() > 0.5 else 0