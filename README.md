# Coin Classification Application

This project is a machine learning-based application designed to classify images of coins into different categories such as pennies, nickels, dimes, and quarters. The project is implemented in Python using PyTorch and supports training, validation, and inference workflows. Below is an overview of the application's architecture, technologies, and functionality.

---

## Features
- **Model Training**: Train a ResNet-18 model on custom datasets.
- **Model Evaluation**: Evaluate the trained model on a validation dataset with metrics such as accuracy, precision, recall, and F1 score.
- **Inference**: Classify coin images with the trained model.
- **Visualization**: Display metrics via bar charts, box plots, and scatter plots.
- **Early Stopping**: Stop training if the validation accuracy does not improve after a set number of epochs.
- **Data Handling**: Automatically splits datasets into training and validation subsets.

---

## Technologies Used

### **Programming Language**
- **Python**: The core language for development, leveraging its rich ecosystem of libraries for machine learning and data manipulation.

### **Frameworks and Libraries**
- **PyTorch**: For model development, training, and inference.
- **TorchVision**: Provides pre-trained models and utilities for image transformations.
- **Scikit-learn**: Used for calculating precision, recall, and F1 scores.
- **Matplotlib**: For visualizing model metrics.
- **Pandas**: To handle tabular data for metric analysis.

### **Data Handling and Augmentation**
- **ImageFolder**: For dataset organization and loading.
- **Transformations**: Data augmentation techniques include resizing, flipping, rotating, cropping, and color jittering to improve model generalization.

### **Infrastructure**
- **Console Application**: A command-line interface for managing training, evaluation, and inference workflows.

### **Visualization Tools**
- Bar, box, and scatter plots for metric evaluation.

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip (Python package manager)
- PyTorch and TorchVision installed (compatible with your system's CUDA setup)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/coin-classifier.git
   cd coin-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the console application:
   ```bash
   python app.py
   ```
2. Select from the following options in the menu:
   - **Run Existing Model on Sample Data**: Use a pre-trained model for inference.
   - **Display Model Metrics**: Visualize training/validation metrics.
   - **Train a New Model**: Train a new model from scratch.
   - **Exit**: Close the application.

---

## Future Improvements
- **Cloud Integration**: Deploy the model using AWS Lambda or SageMaker for serverless inference.
- **Model Optimization**: Use techniques like quantization or model pruning for faster inference.
- **Web Interface**: Add a frontend for ease of use.
- **Advanced Metrics**: Track additional metrics like ROC-AUC.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- The PyTorch community for excellent tools and documentation.
- Scikit-learn for providing robust metric calculations.
- Open-source contributors who maintain visualization libraries like Matplotlib and Pandas.

