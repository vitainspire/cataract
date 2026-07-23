import os
import time
import torch
import torch.nn as nn
import timm
import psutil

# Configuration
CHECKPOINT_FILES = [
    r"E:\Downloads\suyog17_colorblind_exp1.pth",
    r"E:\Downloads\suyog17_colorblind_exp2.pth",
    r"E:\Downloads\suyog17_colorblind_exp3.pth"
]
NUM_CLASSES = 3
device = torch.device('cpu') # Force CPU for local analysis

print(f"--- MODEL ANALYSIS SCRIPT ---")
print(f"System RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

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

models = []
start_time = time.time()

for i, ckpt in enumerate(CHECKPOINT_FILES):
    print(f"\nLoading Model {i+1} ({os.path.basename(ckpt)})...")
    if not os.path.exists(ckpt):
        print(f"ERROR: Cannot find {ckpt}")
        continue
        
    t0 = time.time()
    try:
        m = EyeDiseaseModel(NUM_CLASSES)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
        t1 = time.time()
        print(f"-> Success! Load time: {t1-t0:.2f} seconds")
        print(f"-> RAM Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024**2):.2f} MB")
    except Exception as e:
        print(f"-> FAILED: {str(e)}")

print(f"\nTotal Load Time: {time.time() - start_time:.2f} seconds")

if len(models) > 0:
    print("\n--- INFERENCE PERFORMANCE TEST ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Warmup
    print("Warming up models...")
    with torch.no_grad():
        for m in models: m(dummy_input)
    
    print("Running timed inference...")
    t0 = time.time()
    with torch.no_grad():
        for m in models: m(dummy_input)
    t1 = time.time()
    
    print(f"-> Total Ensemble Inference Time: {t1-t0:.2f} seconds")
    print("Model analysis complete!")
