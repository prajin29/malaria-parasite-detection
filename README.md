# Malaria Parasite Detection Using EfficientNetB0

This project uses transfer learning with EfficientNetB0 to detect malaria from thin blood smear images.

## Try the deployed app here..

* https://malaria-parasite-detection-model.streamlit.app/

## 📁 Dataset

* Cell Images for Malaria Detection
* Contains images of infected and uninfected cells under microscope.

## 🧠 Model

* Used **EfficientNetB0** (pretrained on ImageNet).
* Added data augmentation and a custom classification head.

## 📊 Results

* Accuracy: ~**98-99%** on validation data.
* Trained for 30 epochs using Adam optimizer.

## 🖼 Sample Prediction

* Upload a microscope image of blood cell.
* Model predicts whether it's **Parasitized** or **Uninfected**.

## 🚀 **NEW: Streamlit Web Application**

### **Features:**
- **Beautiful Web UI** for easy malaria detection
- **Image Upload** with drag & drop support
- **Real-time Analysis** using your trained model
- **Professional Results Display** with risk assessment
- **Responsive Design** that works on all devices

## 🔧 **Requirements**
- Python 3.8+
- TensorFlow 2.20.0+
- Streamlit 1.28.0+
- OpenCV (headless version for cloud deployment)
- Pillow
- NumPy 1.24.0+
- Pandas 2.0.0+
