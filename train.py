import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ResNet50

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR = '/data/ILSVRC2012/train'
VAL_DIR   = '/data/ILSVRC2012/val'
CHECKPOINT = None  # set to e.g. 'checkpoint_epoch5.pth' to resume training
NUM_EPOCHS = 30
BATCH_SIZE = 64
LR = 0.1        # SGD with 0.1 is standard for training ResNet50 from scratch
WORKERS = 16

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Transforms ────────────────────────────────────────────────────────────────
class TrainTransform:
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
        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW


class ValTransform:
    # No random augmentations for validation — just resize, crop, normalize
    def __call__(self, img):
        img = self._resize(img, 256)
        img = self._center_crop(img, 224)
        return self._to_tensor(img)

    def _resize(self, img, size):
        w, h = img.size
        if w < h:
            return img.resize((size, int(h * size / w)), Image.BILINEAR)
        return img.resize((int(w * size / h), size), Image.BILINEAR)

    def _center_crop(self, img, size):
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _to_tensor(self, img):
        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)


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


# ── Data ──────────────────────────────────────────────────────────────────────
print(f'Loading train dataset from {TRAIN_DIR}')
train_dataset = ImageNetDataset(root=TRAIN_DIR, transform=TrainTransform())
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=WORKERS,  # parallel CPU workers that load/augment images while the GPU trains
    pin_memory=True       # keeps loaded batches in pinned RAM for faster GPU transfers
)

print(f'Loading val dataset from {VAL_DIR}')
val_dataset = ImageNetDataset(root=VAL_DIR, transform=ValTransform())
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=WORKERS, pin_memory=True
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = ResNet50()

# All layers are trainable — no pretrained weights, training from scratch
for param in model.parameters():
    param.requires_grad = True

model = nn.DataParallel(model)
model = model.to(device)

# ── Optimizer & loss ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
# SGD + momentum is standard for training ResNet from scratch — Adam tends to
# generalize worse on ImageNet when starting from random weights
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
# Reduce LR by 10x at epochs 10 and 20 — standard schedule for 30-epoch runs
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

# ── Resume from checkpoint ────────────────────────────────────────────────────
# Restores the model weights and optimizer state so training continues exactly
# where it left off. Set CHECKPOINT in config to resume from a previous run.
start_epoch = 0
if CHECKPOINT and os.path.isfile(CHECKPOINT):
    print(f'Resuming from {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch']
    print(f'Resumed at epoch {start_epoch}, loss {ckpt["loss"]:.4f}, val acc {ckpt["val_acc"]:.2f}%')

# ── Training loop ─────────────────────────────────────────────────────────────
print(f'Training from epoch {start_epoch + 1} to {start_epoch + NUM_EPOCHS}')
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):

    # ── Train ─────────────────────────────────────────────────────────────────
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
    scheduler.step()  # decay LR at epochs 10 and 20

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed during validation — saves memory and time
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f'Epoch {epoch + 1} done. Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Save model + optimizer + scheduler state so training can be resumed later
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss,
        'val_acc': val_acc,
    }, f'checkpoint_epoch{epoch + 1}.pth')
    print(f'Saved checkpoint_epoch{epoch + 1}.pth')

print('Training complete.')
