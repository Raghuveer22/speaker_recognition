# Speaker Recognition using Neural Networks

Welcome to the `speaker-recognition` project! This repository contains a comprehensive workflow for building a robust speaker identification system using deep learning. The system is designed to classify which speaker is speaking from a set of known individuals, leveraging advanced audio processing and neural network architectures.

---

## **Dataset**

- **Source:** [Speaker Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset)[^4]
- **Description:** Contains 16 kHz PCM audio samples from multiple speakers, including both clean speech and background noise samples. The dataset is organized by speaker and includes folders for background noise augmentation.

---

## **Project Objectives**

- **Data Preparation:** Load, organize, and preprocess audio samples; augment training data with realistic background noise to improve model robustness.
- **Feature Extraction:** Transform raw audio into frequency-domain representations using Fast Fourier Transform (FFT) to capture speaker-specific characteristics.
- **Modeling:** Design and train a Convolutional Neural Network (CNN) with residual blocks for effective speaker classification.
- **Experimentation:** Provide interactive tools for hyperparameter tuning (layers, filters, activation functions) using Jupyter widgets.
- **Evaluation \& Visualization:** Assess model performance with validation accuracy and visualize feature space separation using t-SNE plots[^1][^2].

---

## **Implementation Overview**

### **1. Data Handling \& Augmentation**

- **Organization:** Audio samples are copied and structured into `audio` (speaker data) and `noise` (background noise) directories.
- **Noise Augmentation:**
    - Noise samples are resampled to 16 kHz and split into 1-second chunks.
    - During training, random noise is amplitude-matched and added to clean audio to simulate real-world conditions.
- **Dataset Splitting:** The data is split into training and validation sets (default: 90% train, 10% validation).


### **2. Audio Preprocessing**

- **Audio Loading:** WAV files are decoded and normalized to a fixed sampling rate.
- **Noise Addition:** Custom functions add scaled background noise to each audio sample during training.
- **FFT Transformation:** Each 1-second audio sample is converted to its frequency-domain representation (spectrogram) using FFT, retaining only positive frequencies for analysis[^1][^2].


### **3. Model Architecture**

- **Input:** FFT spectrograms of shape `(8000, 1)` (for 1-second, 16 kHz audio).
- **CNN with Residual Blocks:**
    - Multiple 1D convolutional layers with residual (skip) connections for improved gradient flow and stability.
    - Each block: Conv1D → BatchNorm → ReLU → Skip connection → MaxPooling.
    - Configurable number of blocks, filters, and convolutions per block.
- **Dense Layers:** Flattened features pass through fully connected layers with non-linear activations.
- **Output:** Softmax layer for multiclass speaker classification.


### **4. Training \& Hyperparameter Tuning**

- **Callbacks:** Early stopping and best-model checkpointing.
- **Interactive Tuning:** Jupyter widgets allow users to sweep over model parameters (blocks, filters, activations) and visualize validation accuracy trends.
- **Example Results:** With 2 residual blocks, validation accuracy can reach ~84% after a single epoch[^2].


### **5. Evaluation \& Visualization**

- **Testing:** Inference code enables playback of audio samples with true and predicted labels for qualitative assessment.
- **t-SNE Visualization:** High-dimensional FFT features are projected to 2D using t-SNE, showing clear clustering by speaker, confirming model effectiveness[^1][^2].

---

## **How to Run**

1. **Environment Setup:**
    - Python 3.11+
    - TensorFlow, Keras, NumPy, Matplotlib, scikit-learn, ipywidgets
    - Jupyter Notebook environment (Kaggle or local)
2. **Dataset Preparation:**
    - Download the dataset from Kaggle and place the `16000_pcm_speeches` directory in your working directory.
3. **Notebook Execution:**
    - Run the cells in order to:
        - Organize and preprocess data
        - Augment with noise
        - Extract FFT features
        - Build and train the CNN
        - Evaluate and visualize results
4. **Interactive Exploration:**
    - Use the provided widgets to experiment with model hyperparameters and observe their impact on validation accuracy.

---

## **Key Functions \& Components**

- `safe_copy(src, dst)`: Utility to copy files/directories safely.
- `add_noise(audio, noises, scale)`: Adds amplitude-matched random noise to audio.
- `audio_to_fft(audio)`: Converts time-domain audio to frequency-domain (spectrogram).
- `build_model(input_shape, num_classes, filters_list, conv_num_list, activation)`: Constructs the CNN with residual blocks.
- `run_parameter_sweep(...)`: Interactive function for hyperparameter exploration.
- t-SNE visualization: Visualizes learned feature space for qualitative assessment.

---

## **Results \& Insights**

- **Robustness:** Noise augmentation significantly improves generalization to real-world noisy conditions.
- **Model Performance:** Residual CNN architecture achieves strong validation accuracy, with effective clustering of speakers in feature space.
- **Interpretability:** t-SNE plots reveal clear separation between speakers, validating the model’s discriminative power.

---

## **References**

- [Speaker Recognition Dataset on Kaggle][^4]
- [Course Project Slides and Report][^1]
- [Implementation Notebook][^2]

---

## **Acknowledgements**

This project is based on coursework from the Department of Computer Science and Engineering, IIT Guwahati, and leverages open-source datasets and libraries.

---

**For questions or contributions, please open an issue or pull request. Happy experimenting!**

<div style="text-align: center">⁂</div>

[^1]: Course-Project-DA-623.pdf

[^2]: speaker-recog-latest.ipynb

[^3]: DA623-Winter-2025-_-Course_Project_Assignment_Guidelines.pdf

[^4]: speaker-recognition-dataset

