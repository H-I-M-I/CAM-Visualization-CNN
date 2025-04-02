# Cats vs. Dogs Classification with Class Activation Maps (CAMs)

This project implements a **binary image classification** model to distinguish between cats and dogs using **Convolutional Neural Networks (CNNs)**. Additionally, it applies **Class Activation Maps (CAMs)** to visualize which parts of an image influence the classification decision.

## Dataset
The model is trained on the **Cats vs. Dogs** dataset, which is available through [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs). The dataset is split into:
- **Training Set:** 80%
- **Validation Set:** 10%
- **Test Set:** 10%

## Technologies Used
- **TensorFlow & Keras** – For deep learning and CNN implementation.
- **TensorFlow Datasets (tfds)** – For dataset loading.
- **OpenCV (cv2)** – For image processing.
- **Matplotlib** – For data visualization.

## Implementation Steps
### 1. Load and Preprocess the Dataset
- Load dataset using `tfds.load()`.
- Normalize pixel values and resize images to **300x300**.
- Shuffle and batch the data for efficient training.

### 2. Build the CNN Model
- Implement a **sequential CNN model** with layers:
  - **Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D**
- Use a **sigmoid activation function** for binary classification.

### 3. Train and Evaluate the Model
- Train the model on the preprocessed dataset.
- Evaluate performance using validation and test sets.

### 4. Apply Class Activation Maps (CAMs)
- Generate **Grad-CAM heatmaps** to visualize model focus.
- Overlay heatmaps on input images to highlight important regions.

### 5. Display Results
- Show test images with predicted class labels.
- Display **CAM heatmaps** to understand model decision-making.

## Output
- A trained CNN model capable of classifying **cats and dogs**.
- **Class Activation Maps (CAMs)** to visualize the influential regions of an image.
- Performance metrics for model evaluation.



