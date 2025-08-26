#Training on subset of dataset
import os
import json
import torch
import pickle 
import random
from PIL import Image
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
from model.model import EncoderCNN, DecoderRNN
from model.vocab import Vocabulary
import matplotlib.pyplot as plt
# ---------------------- Dataset Class ----------------------
class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, vocab, transform=None, max_images=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Select a subset of images
        self.selected_ids = set(img['id'] for img in data['images'][:max_images])
        self.image_id_to_file = {img['id']: img['file_name'] for img in data['images'] if img['id'] in self.selected_ids}

        # One caption per image
        used_ids = set()
        self.captions = []
        self.image_ids = []
        for ann in data['annotations']:
            if ann['image_id'] in self.selected_ids and ann['image_id'] not in used_ids:
                self.captions.append(ann['caption'])
                self.image_ids.append(ann['image_id'])
                used_ids.add(ann['image_id'])

        self.vocab.build_vocab(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, self.image_id_to_file[image_id])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Numericalize caption
        caption_idx = [self.vocab.stoi["<SOS>"]]
        caption_idx += self.vocab.numericalize(caption)
        caption_idx.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(caption_idx)

# ---------------------- Collate ----------------------
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions

# ---------------------- Training Function ----------------------
def main():
    # Paths
    root_folder = "train2017"
    annotation_file = "annotations/captions_train2017.json"

    # Hyperparams
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    learning_rate = 3e-4
    batch_size = 64
    num_epochs = 10
    freq_threshold = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Dataset
    vocab = Vocabulary(freq_threshold)
    full_dataset = CocoDataset(root_folder, annotation_file, vocab, transform, max_images=10000)

    # Split into train and val
    val_size = int(0.2 * len(full_dataset))  # 20%
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate(pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx))

    # Model
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        total_train_loss = 0

        for imgs, captions in loop:
            imgs, captions = imgs.to(device), captions.to(device)
            features = encoder(imgs)
            outputs = decoder(features, captions)

            outputs = outputs[:, :captions.size(1)-1, :]
            targets = captions[:, 1:]

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())
        train_losses.append(total_train_loss / len(train_loader))
        # --- Validation ---
        encoder.eval()
        decoder.eval()
        total_val_loss = 0

        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs, captions = imgs.to(device), captions.to(device)
                features = encoder(imgs)
                outputs = decoder(features, captions)
                outputs = outputs[:, :captions.size(1)-1, :]
                targets = captions[:, 1:]
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"📊 Epoch {epoch+1} Summary → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder.state_dict(), "model/best_encoder.pth")
            torch.save(decoder.state_dict(), "model/best_decoder.pth")

    # Final save
    torch.save(encoder.state_dict(), "model/encoder.pth")
    torch.save(decoder.state_dict(), "model/decoder.pth")
    with open("model/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("✅ Training complete. Best model saved as 'best_encoder.pth' & 'best_decoder.pth'")
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")  # Save plot as PNG
    plt.show()

if __name__ == "__main__":
    main()
