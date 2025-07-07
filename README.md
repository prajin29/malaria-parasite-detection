# Malaria Parasite Detection Using EfficientNetB0

This project uses transfer learning with EfficientNetB0 to detect malaria from thin blood smear images.

## 📁 Dataset
- [Cell Images for Malaria Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- Contains images of infected and uninfected cells under microscope.

## 🧠 Model
- Used **EfficientNetB0** (pretrained on ImageNet).
- Added data augmentation and a custom classification head.

## 📊 Results
- Accuracy: ~**98-99%** on validation data.
- Trained for 30 epochs using Adam optimizer.

## 🖼 Sample Prediction
- Upload a microscope image of blood cell.
- Model predicts whether it's **Parasitized** or **Uninfected**.

## 🚀 Future Work
- Add GradCAM visualization.
- Deploy using Streamlit or Flask.
