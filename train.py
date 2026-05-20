import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = '/data/ILSVRC2012/train'
WEIGHTS_FILE = 'resnet50_pretrained.pth'  # pretrained weights to load on first run
CHECKPOINT = None  # set to e.g. 'checkpoint_epoch5.pth' to resume training
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
WORKERS = 8
UNFREEZE_LAYER3 = False

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Transforms ────────────────────────────────────────────────────────────────
class ImageTransform:
    def __call__(self, img):
        img = self._resize(img, 256)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.rotate(random.uniform(-45, 45))
        img = self._color_jitter(img)
        img = self._center_crop(img, 224)
        return self._to_tensor(img)

    def _resize(self, img, size):
        # Resize shortest side to `size`, keep aspect ratio
        w, h = img.size
        if w < h:
            return img.resize((size, int(h * size / w)), Image.BILINEAR)
        return img.resize((int(w * size / h), size), Image.BILINEAR)

    def _center_crop(self, img, size):
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _color_jitter(self, img):
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
        return img

    def _to_tensor(self, img):
        # Convert PIL image to normalized CHW tensor
        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        classes = sorted(os.listdir(root))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # projection shortcut if dimensions differ

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = self.relu(self.bn2(self.conv2(out))) # 3x3 convolve
        out = self.bn3(self.conv3(out))            # 1x1 expand

        out += identity  # skip connection — adds original input back (the residual)
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem — takes raw 3-channel image, outputs 64 channels at 1/4 spatial size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 layer groups — channel sizes and block counts come from the paper
        self.layer1 = self._make_layer(64,   64,  256, blocks=3)
        self.layer2 = self._make_layer(256, 128,  512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride=1):
        # First block always needs a downsample because channels change
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        layers = [BottleneckBlock(in_channels, mid_channels, out_channels, stride=stride, downsample=downsample)]
        # Remaining blocks have matching dimensions so no downsample needed
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# ── Data ──────────────────────────────────────────────────────────────────────
print(f'Loading dataset from {DATA_DIR}')
train_dataset = ImageNetDataset(root=DATA_DIR, transform=ImageTransform())
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=WORKERS,  # parallel CPU workers that load/augment images while the GPU trains
    pin_memory=True       # keeps loaded batches in pinned RAM for faster GPU transfers
)

# ── Setup model ───────────────────────────────────────────────────────────────
model = ResNet50()

# Load pretrained weights — weight file key names match our layer names exactly
if os.path.isfile(WEIGHTS_FILE):
    print(f'Loading pretrained weights from {WEIGHTS_FILE}')
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location='cpu'))

# Freeze all layers — keeps the pretrained weights intact and speeds up training
# since gradients aren't computed for frozen layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the deepest layers so we fine-tune just the high-level features.
# layer4 is the last convolutional block; fc is the final classifier.
# Set UNFREEZE_LAYER3 = True in config to also fine-tune the block before layer4.
if UNFREEZE_LAYER3:
    for param in model.layer3.parameters():
        param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = nn.DataParallel(model)
model = model.to(device)

# ── Optimizer & loss ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
# Only pass unfrozen parameters to the optimizer — no point updating frozen ones
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ── Resume from checkpoint ────────────────────────────────────────────────────
# Restores the model weights and optimizer state so training continues exactly
# where it left off. Set CHECKPOINT in config to resume from a previous run.
start_epoch = 0
if CHECKPOINT and os.path.isfile(CHECKPOINT):
    print(f'Resuming from {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f'Resumed at epoch {start_epoch}, loss {ckpt["loss"]:.4f}')

# ── Training loop ─────────────────────────────────────────────────────────────
print(f'Training from epoch {start_epoch + 1} to {start_epoch + NUM_EPOCHS}')
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            # Print average loss every 100 steps so you can monitor progress mid-epoch
            print(f'Epoch [{epoch + 1}] Step [{i + 1}/{len(train_loader)}] Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    epoch_loss = loss.item()
    print(f'Epoch {epoch + 1} done. Loss: {epoch_loss:.4f}')

    # Save model + optimizer state after each epoch so training can be resumed later
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f'checkpoint_epoch{epoch + 1}.pth')
    print(f'Saved checkpoint_epoch{epoch + 1}.pth')

print('Training complete.')
