import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
from segmen import Segmentor
from coco_seg_dataset import CocoSegDataset
from tqdm import tqdm
import os

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    num_classes = 91
    batch_size = 2
    epochs = 5
    lr = 1e-4

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CocoSegDataset(
        image_dir="train2017",
        mask_dir="masks_train2017",
        ann_file="annotations/instances_train2017.json",
        transform=transform,
        target_size=(128, 128),
        max_samples=1000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,        # Workers cause re-imports
        pin_memory=True
    )

    model = Segmentor(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 Starting segmentation training...")
    print(f"Using device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} complete — Average Loss: {avg_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/segmentor.pth")
    print("✅ segmentor.pth saved.")


if __name__ == "__main__":
    main()
