# Handwritten Letter Classifier

This is my first custom neural network project using TensorFlow and Keras, trained on my own handwritten letters.

## What I Did:

- Collected and labeled 180 grayscale images (20 samples each of A–I)
- Built and trained a Convolutional Neural Network (CNN) to classify the letters
- Evaluated the model using confusion matrix and accuracy plots

## What I Learned:

- How to prepare and label a small custom dataset
- Basics of CNNs and their architecture
- How to split and normalize image data manually

## Model Architecture

- Conv2D (3×3) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Conv2D (2×2) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Conv2D (2×3) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Flatten
- Dense (32) – ReLU
- Dropout (0.2)
- Dense (64) – ReLU
- Dropout (0.3)
- Dense (9) – Sigmoid (one for each letter class)

- Loss: sparse_categorical_crossentropy
- Optimizer: adam
- Epochs: 50

## Evaluation:

- Train/Test Split: 80 % / 20 %

- ### Accuracy:
- Train: ~90 %
- Test: ~91 %

- Confusion matrix shows results across all classes

## Visualizations

- Train vs. Test accuracy plot
- Confusion matrix
- Example predictions using matplotlib

## Dataset

- Custom dataset: grayscale images (28×28)
- 9 classes: A, B, C, D, E, F, G, H, I
- Saved as directory-based structure, loaded with image_dataset_from_directory

## Next Steps

- Improve model generalization with more diverse samples
- Expand to more characters (full alphabet or digits)
- Build a GUI for real-time handwriting input and classification
