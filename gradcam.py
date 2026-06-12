import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model import ResNet50

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH = '/data/ILSVRC2012/train/n03347037/n03347037_1567.JPEG'
CHECKPOINT = 'checkpoint_epoch30.pth'  # change to whichever checkpoint you want to inspect
OUTPUT_FILE = 'gradcam.png'

# ── Load model ────────────────────────────────────────────────────────────────
# Load on CPU so it doesn't interfere with training running on the GPU
device = torch.device('cpu')
model = ResNet50()
ckpt = torch.load(CHECKPOINT, map_location=device)

# Strip the 'module.' prefix added by DataParallel during training
state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# ── Grad-CAM hooks ────────────────────────────────────────────────────────────
# These intercept the forward and backward pass through layer4 to capture
# activations and gradients, which we use to build the heatmap
gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)
    output.register_hook(save_gradient)

model.layer4.register_forward_hook(forward_hook)

# ── Preprocess image ──────────────────────────────────────────────────────────
img = Image.open(IMAGE_PATH).convert('RGB')
img_resized = img.resize((224, 224))
arr = np.array(img_resized, dtype=np.float32) / 255.0
arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
tensor = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)  # HWC -> CHW -> NCHW

# ── Forward + backward pass ───────────────────────────────────────────────────
output = model(tensor)
pred_class = output.argmax().item()
model.zero_grad()
# Backprop from the predicted class score — tells us which parts of layer4
# were most important for that prediction
output[0, pred_class].backward()

# ── Compute heatmap ───────────────────────────────────────────────────────────
grads = gradients[0].squeeze().numpy()        # [2048, 7, 7]
acts = activations[0].squeeze().detach().numpy()  # [2048, 7, 7]
weights = grads.mean(axis=(1, 2))             # average gradient per channel
cam = np.zeros(acts.shape[1:])
for i, w in enumerate(weights):
    cam += w * acts[i]
cam = np.maximum(cam, 0)                      # ReLU — only keep positive activations
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize to [0, 1]
cam = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))

# ── Overlay and save ──────────────────────────────────────────────────────────
heatmap = cm.jet(cam)[:, :, :3]
original = np.array(img_resized) / 255.0
overlay = 0.5 * original + 0.5 * heatmap

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(original);        axes[0].set_title('Original');  axes[0].axis('off')
axes[1].imshow(cam, cmap='jet'); axes[1].set_title('Grad-CAM');  axes[1].axis('off')
axes[2].imshow(overlay);         axes[2].set_title('Overlay');   axes[2].axis('off')
plt.suptitle(f'Predicted class index: {pred_class} | Checkpoint: {CHECKPOINT}')
plt.tight_layout()
plt.savefig(OUTPUT_FILE)
print(f'Saved {OUTPUT_FILE}')
