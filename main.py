import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# モデルの定義
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], -1)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
model = SimpleMLP().to(device)
model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device))
model.eval()

st.title("手書き数字分類 (MNIST)")

uploaded_file = st.file_uploader("28x28の手書き数字画像（白地に黒字）をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # グレースケール化
    image = ImageOps.invert(image)  # 白黒反転（MNIST形式に合わせる）
    image = image.resize((28, 28))  # サイズ調整

    st.image(image, caption="入力画像", width=150)

    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)

    st.write(f"**予測された数字:** {predicted.item()}")
    st.bar_chart(probabilities.squeeze().cpu().numpy())
