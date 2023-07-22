import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os
import re
from colorama import Fore, Style, init

from torchvision.models import ResNet18_Weights

def id_coins(sample_dir, model_dir, train_dir):
    # Initialize colorama
    init()

    # Set device for computation (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the ResNet18 model architecture
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Get the number of input features for the final fully connected layer
    num_features = model.fc.in_features

    # Define the number of classes as the number of folders in the training directory
    num_classes = len(os.listdir(train_dir))
    # Replace the final layer of the model to match our number of classes
    model.fc = nn.Linear(num_features, num_classes)

    # Get all the saved model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

    # Sort model files by the epoch number (assumed to be in the file name)
    model_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    # Get the latest model file
    latest_model_file = model_files[-1]

    # Define the path to the latest model file
    model_path = os.path.join(model_dir, latest_model_file)

    # Load the latest model's parameters
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Move the model to the computation device
    model = model.to(device)
    # Set the model to evaluation mode (turns off dropout, etc.)
    model.eval()

    # Print out the loaded model's metrics
    print("Loaded model from epoch {} with accuracy {}, precision {}, recall {}, and F1 score {}.".format(
        checkpoint['epoch'], checkpoint['accuracy'], checkpoint['precision'], checkpoint['recall'], checkpoint['f1_score']))

    # Define the classes for classification
    class_names = ['penny', 'nickel', 'quarter']

    # Define the transform to apply to the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the coin names to check in the file name
    coin_names = ['dime', 'nickel', 'penny', 'cent', 'quarter']

    # For each image in the sample directory
    for file_name in os.listdir(sample_dir):
        # Check if the file is an image
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Define the path to the image file
            image_path = os.path.join(sample_dir, file_name)
            # Open the image file
            image = Image.open(image_path)

            # Apply the transform to the image and add an extra dimension
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Without computing gradients (for efficiency and memory saving)
            with torch.no_grad():
                # Make a forward pass through the model and get the output
                output = model(image_tensor)
                # Calculate probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)
                # Get the predicted class
                _, predicted = torch.max(output, 1)

            # Check confidence level
            confidence = probabilities[0][predicted.item()].item()
            if confidence < 0.6:
                print(f'{Fore.YELLOW}{file_name} is unknown with confidence of: {confidence}{Style.RESET_ALL}')
            else:
                # Print out the results with color coded confidence levels
                predicted_class = class_names[predicted.item()]
                # Convert both the file name and predicted class to lower case for case-insensitive comparison
                file_name_lower = file_name.lower()
                predicted_class_lower = predicted_class.lower()

                # Check if any of the coin names are in the file name
                if any(coin_name in file_name_lower for coin_name in coin_names):
                    # If the predicted class is in the file name, print in green; otherwise, print in red
                    if predicted_class_lower in file_name_lower:
                        print(f'{Fore.GREEN}{file_name} is a {predicted_class} with confidence of: {confidence}{Style.RESET_ALL}')
                    else:
                        print(f'{Fore.RED}{file_name} is a {predicted_class} with confidence of: {confidence}{Style.RESET_ALL}')
                else:
                    # If none of the coin names are in the file name, print without color
                    print(f'{file_name} is a {predicted_class} with confidence of: {confidence}')
