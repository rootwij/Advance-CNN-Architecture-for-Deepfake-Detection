# 🧠 Deepfake Detection using CNN Architectures

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository presents multiple Convolutional Neural Network (CNN) models—ResNet50, VGG19, EfficientNet, and XceptionNet—for detecting deepfake images. Each model leverages transfer learning and is trained on a real vs. fake image dataset sourced from Kaggle.

---

## 📁 Repository Structure

```
📦 deepfake-detection/
├── EfficientNet.ipynb           # EfficientNet model notebook
├── RESNET.ipynb                 # ResNet model (baseline)
├── RESNET_updated.ipynb         # Fine-tuned ResNet model
├── resnet.py                    # Script version of ResNet pipeline
├── RDL_Project_RestNet.ipynb    # Research/Report version using ResNet
├── VGG19.ipynb                  # VGG19 model notebook
├── vgg19.py                     # Script version of VGG19 pipeline
├── Xceptionet.ipynb             # XceptionNet model notebook
├── kaggle.json                  # Kaggle API token (required for dataset download)
└── README.md                    # Project documentation
```

---

## 📦 Dataset

- **Name**: [Deepfake Dataset](https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset)
- **Source**: Kaggle
- **Structure**:
  ```
  /train/
      real/
      fake/
  /valid/
      real/
      fake/
  ```

To download, place your `kaggle.json` in the root directory and run:

```bash
!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d tusharpadhy/deepfake-dataset
!unzip deepfake-dataset.zip
```

---

## 🏗️ Models

| Model        | Backbone       | Params Tuned | Accuracy (Val) | Notes                     |
|--------------|----------------|---------------|----------------|---------------------------|
| ResNet50     | torchvision     | Final FC, layer4 | ~92%           | Supports fine-tuning      |
| VGG19        | torchvision     | Classifier layers | ~91%           | Lightweight, simple       |
| EfficientNet | timm            | Top layers    | ~94%           | Best performance           |
| XceptionNet  | keras (ported)  | Custom head   | ~94%            | Depthwise separable CNN   |

All models are adapted for binary classification using `BCEWithLogitsLoss`.

---

## 🧪 Training Pipeline

- Data loading via `torchvision.datasets.ImageFolder`
- Augmentation via `transforms.Compose` (flip, crop, rotate)
- Training loop with:
  - Transfer learning
  - Fine-tuning support
  - GPU support
- Evaluation:
  - Accuracy, loss curves
  - Confusion matrix
  - Classification report

---

## 🚀 How to Use

1. Clone this repo:

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

2. Install dependencies:

```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn kaggle
```

3. Download the dataset (see Dataset section above)

4. Run a model notebook (`.ipynb`) in Jupyter or Colab

---

## 📊 Results Visualization

The notebooks include:
- Training/validation loss & accuracy plots
- Confusion matrix heatmaps
- Visual examples of model predictions

---

## ✅ Features

- [x] Pretrained model transfer learning
- [x] Fine-tuning support
- [x] Evaluation metrics & visualizations
- [x] Modular `.py` scripts for automation
- [x] Kaggle API integration

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Kaggle](https://www.kaggle.com/)
- [HuggingFace Timm](https://github.com/huggingface/pytorch-image-models)
