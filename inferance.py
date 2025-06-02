import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn as nn
from torchvision import models

# --- CONFIG ---
IMAGE_SIZE   = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT   = "best_utkface_resnet.pt"

# --- MODEL DEFINITION (must match your training script) ---
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # load the ResNet‐18 backbone (weights=None since we load state_dict)
        backbone = models.resnet18(weights=None)
        # strip off the final fc layer → this is your feature_extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        hidden = backbone.fc.in_features  # should be 512

        # gender head
        self.gender_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 4, 2)
        )
        # age head
        self.age_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 4, 1)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        feats = self.feature_extractor(x)             # [B,512,1,1]
        feats = feats.view(feats.size(0), -1)         # [B,512]
        gender_logits = self.gender_head(feats)       # [B,2]
        age_pred = self.age_head(feats).squeeze(-1)   # [B]
        return gender_logits, age_pred

# --- LOAD MODEL & CHECKPOINT ---
model = MultiTaskResNet().to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state)  # now keys match: feature_extractor.*, gender_head.*, age_head.*
model.eval()

# --- PREPROCESSOR (same resize + normalize as training) ---
preprocess = Compose([
    Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

def predict_image(image):
    """
    Predict gender and age from a PIL Image
    Args:
        image: PIL Image object
    Returns:
        tuple: (gender, age) where gender is "male" or "female" and age is a float
    """
    x = preprocess(image).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    
    with torch.no_grad():
        g_logits, a_pred = model(x)
        
        g_idx = int(g_logits.argmax(dim=-1).item())
        gender = "female" if g_idx == 1 else "male"
        age = float(a_pred.item())
        
    return gender, age
        
