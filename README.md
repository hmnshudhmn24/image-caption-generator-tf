# 🖼️ Image Caption Generator

This project generates captions for images using a deep learning model that combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with an encoder-decoder architecture.

## 🧠 Model Overview

- **Encoder**: CNN (InceptionV3) extracts image features.
- **Decoder**: LSTM-based RNN generates captions based on extracted features.
- **Dataset**: MSCOCO (or small sample for demo).

## 📦 Requirements

```bash
pip install tensorflow numpy matplotlib pillow tqdm
```

## 🚀 How to Run

1. Place your images inside the `images/` folder.
2. Run the script:
```bash
python image_caption_generator.py
```

## 📁 Files

- `image_caption_generator.py` — Main script with model pipeline.
- `README.md` — Project overview and instructions.

## 📌 Notes

- Uses transfer learning (InceptionV3) for encoding.
- Supports generating captions using beam search or greedy decoding.

## 🧾 Output

Example:
> 🖼️ image.jpg ➝ "A dog is playing in the park."