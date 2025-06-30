# GeoVisionAI

# 🌍 Satellite Image Segmentation, Prediction & Natural Event Detection

This project aims to develop deep learning models for satellite image segmentation and natural disaster detection, with a particular focus on **land cover classification** and **landslide prediction**. It consists of three main components:

---

## 🔧 Project Breakdown

### 1. **Multiclass Satellite Image Segmentation (Baseline UNet Model)**
- Utilized a **UNet** deep learning architecture trained on the **DeepGlobe Dataset** (sourced from Kaggle).
- Dataset:
  - 813 training images
  - 172 validation images (merged with test set due to missing masks)
  - 171 test images
- To overcome the lack of masks in validation data, we created a small validation subset from the training data.
- Metric used: **Intersection over Union (IoU)**
- Image size downsized from **2448×2448** to **1024×1024** and **512×512** for efficient training.
- Trained for **100 epochs** using **Adam optimizer** with a learning rate of **0.01**.

---

### 2. **Landslide Detection via Transfer Learning**
- Applied transfer learning using the previously trained UNet baseline model.
- Modified the architecture for **binary classification** (landslide vs. no landslide).
- Froze all layers except the last, which was retrained for landslide mask generation.
- Trained for **50 epochs** on landslide-labelled images from a custom dataset.
- Resulted in effective landslide segmentation masks.

---

### 3. **Binary Image Classification for Landslide Presence**
- Used the pretrained model to classify full satellite images as either:
  - `Landslide`
  - `Non-landslide`
- Achieved a classification accuracy of **~70–75%** on the labeled dataset.

---

## 🛠️ Tools & Libraries
- Python 🐍
- PyTorch 🔥

---

## 📊 Results & Visualizations

### 🟩 Intersection Over Union (IoU) on 60 Test Samples
![IOU](https://user-images.githubusercontent.com/32778343/153116591-9d57737f-ec65-48ff-8db6-dc6a124e2fcc.JPG)

### 🛰️ Real Satellite RGB Images
![RGB](https://user-images.githubusercontent.com/32778343/153116593-32cd15b7-1247-4644-8eac-990465029c38.JPG)

### 🎯 Segmentation Output from UNet
![Segmentation](https://user-images.githubusercontent.com/32778343/153116594-a56d00d5-206e-4326-add6-efc919e63161.JPG)

### 🌋 Landslide Satellite Images from Bijie Dataset
![Landslide Data](https://user-images.githubusercontent.com/32778343/153116595-33407ee6-e110-48db-aa59-00f1f8425837.JPG)

### 🧠 Transfer Learning Results (Landslide Segmentation)
![Landslide Prediction](https://user-images.githubusercontent.com/32778343/153116596-a62314ff-472b-4d71-b4ea-26150caa2ea6.JPG)

### 📈 Accuracy Plot — Baseline vs. Landslide Model
![Accuracy](https://user-images.githubusercontent.com/32778343/153116597-55d20cfb-7945-4085-b421-a2de63343206.JPG)

### 📉 Loss Comparison — Baseline vs. Landslide Model
![Loss](https://user-images.githubusercontent.com/32778343/153116598-2f448421-ac99-4106-8796-88fb0330d995.JPG)

---

## 📌 Future Work
- Incorporate temporal modelling for forecasting future events
- Add attention mechanisms (e.g., SE blocks, transformers)
- Deploy a web-based geospatial dashboard for real-time detection

---

## 📁 Dataset References
- [DeepGlobe Land Cover Classification Dataset (Kaggle)](https://www.kaggle.com/c/deepglobe-land-cover-classification-challenge)
- Bijie-Landslide Dataset (custom curated)

---

## 📜 License
This project is licensed under the MIT License.

---

## 🙌 Acknowledgements
Thanks to the creators of the DeepGlobe dataset and the open-source PyTorch community.

