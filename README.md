# Plant Disease Classification Using EfficientNetB0

This project uses EfficientNetB0 transfer learning to classify plant leaf diseases from the PlantVillage dataset. The model is trained with image augmentation, proper preprocessing, and modern deep learning techniques to achieve high accuracy and strong generalization.

## Features
- Dataset loading & preprocessing
- Image augmentation for robustness
- Transfer learning with EfficientNetB0 (ImageNet weights)
- Frozen base model + custom classification head
- Training with Adam optimizer
- Evaluation using accuracy, confusion matrix, and classification report

## Model Architecture
### 🔹 EfficientNetB0 (pre-trained on ImageNet)
```
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

### 🔹 Custom Classification Head
- Data augmentation
- EfficientNet preprocessing
- Global Average Pooling
- Dropout (0.3)
- Dense output layer with softmax activation

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Google Colab

## Training & Evaluation
- Optimizer: Adam (lr = 1e-3)
- Loss: sparse categorical crossentropy
- Metrics: accuracy

Evaluation includes:
- Validation accuracy
- Confusion matrix
- Classification report
- Balanced accuracy
- Grad-CAM visualizations

## 📁 Project Structure
```
PlantDiseaseClassification/
│
├── notebook.ipynb
├── README.md
│
├── results/
│   ├── accuracy-plot.png
│   ├── loss-plot.png
│   ├── confusion-matrix.png
│   
│
├── model/
│   └── best_model.h5

```

## Future Work
- Fine-tuning deeper EfficientNet layers
- Trying EfficientNetB3/B4 or Vision Transformers
- Ensemble models
- Deployment as mobile/web app
- Adding real-world data
- Using object detection for disease localization

## How to Run
1. Open the notebook in Google Colab
2. Upload dataset or connect Google Drive
3. Run all cells
4. View predictions and visualizations

## Acknowledgements
- PlantVillage Dataset (https://data.mendeley.com/datasets/tywbtsjrjv/1)
- TensorFlow / Keras
- Google Colab
