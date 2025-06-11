# main.py
import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# ――― モデル定義 ――― #
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, 784]
        h = torch.sigmoid(self.fc1(x))
        return self.fc2(h)


# ――― モデル読み込み ――― #
@st.cache_resource(show_spinner=True)
def load_model(weight_path: Path, device: torch.device):
    model = SimpleMLP().to(device)
    try:  # PyTorch ≥ 2.2
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:  # それ以前
        state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ――― 画像前処理 ――― #
transform = T.Compose([
    T.Grayscale(num_output_channels=1),      # 1ch 28×28
    T.Resize((28, 28)),
    T.ToTensor(),                            # [0-1] へ変換
])


def preprocess(img_bytes: bytes, device: torch.device):
    image = Image.open(io.BytesIO(img_bytes)).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]
    return tensor, image


# ――― Streamlit UI ――― #
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("手書き数字分類 (Simple MLP)")

uploaded = st.file_uploader(
    "28×28 px または任意サイズのモノクロ／カラー PNG・JPEG をアップロードしてください",
    type=["png", "jpg", "jpeg"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = Path("modelwithBatch.pth")

if not model_path.exists():
    st.error(f"{model_path} が見つかりません。ファイルを同じフォルダに置いてください。")
    st.stop()

model = load_model(model_path, device)

if uploaded is not None:
    tensor, pil_img = preprocess(uploaded.getvalue(), device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze()

    pred = int(torch.argmax(probs))
    st.image(pil_img.resize((140, 140)), caption=f"予測: {pred}", clamp=True)

    st.subheader("各クラス確率")
    st.bar_chart({str(i): float(probs[i]) for i in range(10)})
else:
    st.info("左のボタンから画像を選択してください。")

