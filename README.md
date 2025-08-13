# Handwritten Letter Classifier

This is my first custom neural network project using TensorFlow and Keras, trained on my own handwritten letters.

## What I Did:

- Collected and labeled 620 grayscale images (20+ samples each of A–Z)
- Built and trained a Convolutional Neural Network (CNN) to classify the letters
- Evaluated the model using confusion matrix and accuracy plots
- Built a GUI with tkinter (used Copilot for difficult things)

## What I Learned:

- How to prepare and label a small custom dataset
- Basics of CNNs and their architecture
- How to split and normalize image data manually
- Basics of tkinter

## Model Architecture

- Conv2D (3×3) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Conv2D (2×2) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Conv2D (2×3) – 32 filters + ReLU
- MaxPooling2D (2×2)
- Flatten
- Dense (64) – ReLU
- Dropout (0.3)
- Dense (64) – ReLU
- Dropout (0.3)
- Dense (26) – Softmax (one for each letter class)

- Loss: sparse_categorical_crossentropy
- Optimizer: adam
- Epochs: 100

## Evaluation:

- Train/Test Split: 80 % / 20 %

- ### Accuracy:
- Train: ~96 %
- Test: ~87 %

- Confusion matrix shows results across all classes

## Visualizations

- Train vs. Test accuracy plot
- Confusion matrix
- Example predictions using matplotlib

## Dataset

- Custom dataset: grayscale images (28×28)
- 26 classes: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- Saved as directory-based structure, loaded with image_dataset_from_directory

## Next Steps

- Improve model generalization with more diverse samples
