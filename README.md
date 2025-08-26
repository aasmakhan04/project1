🖼️ Image Captioning with MS COCO Dataset

This project implements an Image Captioning model using the MS COCO 2017 dataset. The model learns to generate natural language captions for images by combining a CNN encoder (for image features) and an RNN decoder with embeddings (for text generation).

Additionally, the training script supports both training and validation splits, tracks losses, and saves the best-performing model.

🚀 Features
MS COCO 2017 support (captions_train2017.json, captions_val2017.json)
CNN (Encoder) + RNN (Decoder) architecture
Vocabulary built from COCO captions with <SOS>, <EOS>, <PAD> tokens
Training + Validation split with loss tracking
Best model saving (best_encoder.pth, best_decoder.pth)
Training vs Validation loss plot (loss_plot.png)

📂 Project Structure
project/
├── train.py                # Training script
├── model/
│   ├── model.py             # EncoderCNN & DecoderRNN
│   ├── vocab.py             # Vocabulary builder
│   ├── encoder.pth          # Final trained encoder
│   ├── decoder.pth          # Final trained decoder
│   ├── best_encoder.pth     # Best validation encoder
│   ├── best_decoder.pth     # Best validation decoder
│   └── vocab.pkl            # Saved vocabulary
├── annotations/
│   └── captions_train2017.json
├── train2017/               # COCO training images
├── val2017/                 # COCO validation images
└── loss_plot.png            # Training vs Validation loss graph


⚙️ Requirements
Install dependencies using pip:
pip install torch torchvision tqdm pillow matplotlib

▶️ Training
Run the training script:
python train.py
Training and validation losses will be printed per epoch.
Best models will be saved automatically in the model/ directory.
A training-vs-validation loss plot will be generated (loss_plot.png).


📊 Example Training Output
Epoch 1/10 [Training]: 100%|██████████████| 157/157 [00:30<00:00, 5.19it/s, train_loss=3.42]
📊 Epoch 1 Summary → Train Loss: 3.2501 | Val Loss: 3.0812
Epoch 2/10 [Training]: ...


📈 Evaluation Metric
The training uses CrossEntropyLoss (per-token log-loss).
To make it more interpretable, you can compute Perplexity (PPL):
Perplexity = exp(Loss)
Lower perplexity = better captions.


🔮 Future Work
Add Attention-based decoder (Bahdanau/Luong).
Integrate Transformer-based captioning (ViT + GPT style).
Add BLEU score evaluation for captions.
Build a Streamlit app for interactive image captioning.


📝 License
This project is for educational and research purposes.
MS COCO dataset must be separately downloaded and used under its license.
