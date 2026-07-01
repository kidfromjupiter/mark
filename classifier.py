from pathlib import Path

import torch
import torch.nn as nn

DATA_DIR = Path("dataset")
EMPTY_MODEL_PATH = Path("mcq_empty_classifier.pt")
MARK_MODEL_PATH = Path("mcq_mark_type_classifier.pt")
MODEL_PATH = MARK_MODEL_PATH
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEFAULT_CLASSES = ["crossed", "empty", "scribble"]
EMPTY_STAGE_CLASSES = ["empty", "marked"]
MARK_STAGE_CLASSES = ["crossed", "scribble"]


class SimpleMCQCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_trained_model(model_path=MODEL_PATH, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        img_size = checkpoint.get("img_size", IMG_SIZE)
        state_dict = checkpoint["model_state"]
    else:
        classes = DEFAULT_CLASSES
        img_size = IMG_SIZE
        state_dict = checkpoint

    model = SimpleMCQCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, list(classes), int(img_size)


def build_transforms():
    from torchvision import transforms

    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    return train_tfms, val_tfms


class RemappedImageFolder:
    def __init__(self, root, transform, class_map, classes):
        from torchvision import datasets

        self.dataset = datasets.ImageFolder(root, transform=transform)
        self.classes = list(classes)
        self.samples = [
            (path, self.classes.index(class_map[self.dataset.classes[target]]))
            for path, target in self.dataset.samples
            if self.dataset.classes[target] in class_map
        ]

        if not self.samples:
            raise RuntimeError(f"No samples found in {root} for classes: {classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.dataset.loader(path)

        if self.dataset.transform is not None:
            sample = self.dataset.transform(sample)

        return sample, target


def class_counts(dataset):
    counts = {class_name: 0 for class_name in dataset.classes}

    for _, target in dataset.samples:
        counts[dataset.classes[target]] += 1

    return counts


def class_weights(dataset, device):
    counts = class_counts(dataset)
    total = sum(counts.values())
    weights = [
        total / (len(dataset.classes) * counts[class_name])
        for class_name in dataset.classes
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_stage(
    stage_name,
    output_path,
    classes,
    class_map,
    train_tfms,
    val_tfms,
    device,
):
    from torch.utils.data import DataLoader

    train_ds = RemappedImageFolder(DATA_DIR / "train", train_tfms, class_map, classes)
    val_ds = RemappedImageFolder(DATA_DIR / "val", val_tfms, class_map, classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"[{stage_name}] Classes:", train_ds.classes)
    print(f"[{stage_name}] Train counts:", class_counts(train_ds))
    print(f"[{stage_name}] Val counts:", class_counts(val_ds))

    model = SimpleMCQCNN(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)

        model.eval()
        val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_ds)

        print(
            f"[{stage_name}] Epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "img_size": IMG_SIZE,
                },
                output_path,
            )

    print(f"[{stage_name}] Best val acc:", best_val_acc)


def train(device="cuda", stage="all"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --cpu.")

    print("Using:", device)
    train_tfms, val_tfms = build_transforms()

    stages = {
        "empty": {
            "stage_name": "empty",
            "output_path": EMPTY_MODEL_PATH,
            "classes": EMPTY_STAGE_CLASSES,
            "class_map": {
                "empty": "empty",
                "crossed": "marked",
                "scribble": "marked",
            },
        },
        "mark": {
            "stage_name": "mark",
            "output_path": MARK_MODEL_PATH,
            "classes": MARK_STAGE_CLASSES,
            "class_map": {
                "crossed": "crossed",
                "scribble": "scribble",
            },
        },
    }

    stage_names = ["empty", "mark"] if stage == "all" else [stage]

    for stage_name in stage_names:
        train_one_stage(
            train_tfms=train_tfms,
            val_tfms=val_tfms,
            device=device,
            **stages[stage_name],
        )

