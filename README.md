Plant Disease Classification Using EfficientNetB0

This project uses EfficientNetB0 transfer learning to classify plant leaf diseases from the PlantVillage dataset. The model is trained with image augmentation, proper preprocessing, and modern deep learning techniques to achieve high accuracy and strong generalization.

🚀 Features

✔️ Dataset loading & preprocessing

✔️ Image augmentation for robustness

✔️ Transfer learning with EfficientNetB0 (ImageNet weights)

✔️ Frozen base model + custom classification head

✔️ Training with Adam optimizer

✔️ Evaluation using accuracy, confusion matrix, and classification report


Model Architecture

The model uses:

🔹 EfficientNetB0 (pre-trained on ImageNet)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)


Base model is frozen (not trainable)

Only the top layers are trained

🔹 Custom Classification Head

Data augmentation

EfficientNet preprocessing

Global Average Pooling

Dropout (0.3)

Dense output layer with softmax activation

Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Google Colab

Training & Evaluation

The model is trained using:

Optimizer: Adam (lr = 1e-3)

Loss: sparse categorical crossentropy

Metrics: accuracy

Evaluation includes:

Validation accuracy

Confusion matrix

Classification report

Balanced accuracy

Grad-CAM visualizations for model interpretation


Example Outputs

Training accuracy & loss curves

Confusion matrix showing class-level performance

Grad-CAM heatmap highlighting affected regions of leaves

Future Work

Fine-tuning deeper EfficientNet layers

Trying EfficientNetB3 / B4 or Vision Transformers

Using ensemble models

Deploying as a mobile or web app

Expanding dataset with real-world images

Adding object detection for localized disease spots

