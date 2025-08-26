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

### **How to Run Locally:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/prajin29/malaria-parasite-detection.git
   cd malaria-parasite-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** at `http://localhost:8501`

### **How to Use:**
1. **Upload Image:** Drag and drop or browse for a blood cell image
2. **Analyze:** Click the "Analyze Image" button
3. **Get Results:** View infection probability and medical recommendations

## 🌐 **Deploy to Streamlit Cloud (Free)**

### **Step 1: Push to GitHub**
Make sure your `app.py` and `malaria_model.h5` are in your repository.

### **Step 2: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `prajin29/malaria-parasite-detection`
5. Set **Main file path:** `app.py`
6. Click **"Deploy"**

### **Step 3: Access Your App**
Your app will be available at: `https://your-app-name.streamlit.app`

## 📁 **Repository Structure**
```
malaria-parasite-detection/
├── Malaria_Detection.ipynb          # Training notebook
├── malaria_model.h5                 # Trained model
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🔧 **Requirements**
- Python 3.8+
- TensorFlow 2.20.0+
- Streamlit 1.28.0+
- OpenCV (headless version for cloud deployment)
- Pillow
- NumPy 1.24.0+
- Pandas 2.0.0+

## 🎯 **Future Work**
* ✅ Add Streamlit deployment
* Add GradCAM visualization
* Deploy using Flask (alternative)
* Add batch processing for multiple images
* Implement confidence scoring

## 📞 **Support**
If you encounter any issues:
1. Check the requirements.txt for compatible versions
2. Ensure your model file is in the same directory as app.py
3. Open an issue on GitHub with error details

---
**Note:** This tool is for educational and research purposes only. Always consult healthcare professionals for medical diagnosis.
