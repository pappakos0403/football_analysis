import torch

# Modell betöltése
model = torch.load('yolo11s.pt')
model.eval()