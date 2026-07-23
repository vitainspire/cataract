import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# Configuration
CHECKPOINT_FILES = [
    os.getenv("CHECKPOINT_EXP1", r"E:\Downloads\suyog17_colorblind_exp1.pth"),
    os.getenv("CHECKPOINT_EXP2", r"E:\Downloads\suyog17_colorblind_exp2.pth"),
    os.getenv("CHECKPOINT_EXP3", r"E:\Downloads\suyog17_colorblind_exp3.pth")
]
NUM_CLASSES = 3
CLASS_NAMES = {0: "Cataract", 1: "Normal", 2: "Not Eye"}
device = torch.device('cpu')

class EyeDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def condition_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(denoised)
    return cv2.merge((cl, cl, cl))

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def test_image(img_path):
    print(f"\n====================================")
    print(f"Testing image: {img_path}")
    print(f"====================================")

    if not os.path.exists(img_path):
        print(f"ERROR: Image not found at {img_path}")
        return

    print("-> Loading Ensemble Models...")
    models = []
    for ckpt in CHECKPOINT_FILES:
        if os.path.exists(ckpt):
            m = EyeDiseaseModel(NUM_CLASSES)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            models.append(m)
    
    if not models:
        print("ERROR: No checkpoints found!")
        return

    try:
        pil_img = Image.open(img_path).convert('RGB')
        
        # --- REMOVED COLORBLIND CONDITIONING ---
        # The AI now gets to see the true, full-color RGB image!
        
        tensor = val_transform(pil_img).unsqueeze(0).to(device)

        print("-> Running Ensemble Inference...")
        with torch.no_grad():
            outputs = [m(tensor) for m in models]
            probs = [F.softmax(out, dim=1) for out in outputs]
            stacked_probs = torch.stack(probs, dim=0)
            ensemble_probs, _ = torch.max(stacked_probs, dim=0)
            
            probs_np = ensemble_probs.cpu().numpy()[0]
            probs_np = probs_np / np.sum(probs_np)
            
            pred_idx = int(np.argmax(probs_np))
            print(f"\n[ RESULTS ]")
            print(f" - Cataract: {probs_np[0]*100:.2f}%")
            print(f" - Normal:   {probs_np[1]*100:.2f}%")
            print(f" - Not Eye:  {probs_np[2]*100:.2f}%")
            print(f"\n DIAGNOSIS: {CLASS_NAMES[pred_idx]} ({probs_np[pred_idx]*100:.2f}%)")
            
    except Exception as e:
        print(f"ERROR processing image: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the image to test")
    args = parser.parse_args()
    test_image(args.image_path)
