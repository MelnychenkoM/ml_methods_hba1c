import numpy as np
import torch
import os

from models import Lightning1DCNNModel

CHECKPOINT = 'checkpoints/last.ckpt'
FOLDER_PATH = "../../data/train_test_cnn/"

def load_data(folder, filename):
    file_path = os.path.join(folder, filename)
    return np.load(file_path).astype(np.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Lightning1DCNNModel.load_from_checkpoint(CHECKPOINT)
model.to(device)
model.eval()


X_test = load_data(FOLDER_PATH, "X_test.npy")
y_test = load_data(FOLDER_PATH, "y_test.npy")

X_test_tensor = torch.tensor(X_test, device=device)

with torch.no_grad():
    res = model(X_test_tensor)

res_np = res.detach().cpu().numpy().reshape(-1)

np.save("../../data/predict_cnn.npy", np.stack([res_np, y_test], axis=1))
