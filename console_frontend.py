import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch

import build_model
import run_model
import split_data


def run(main_dir, train_dir, val_dir, sample_dir, model_dir):
    while True:
        print("\nPlease select an option:")
        print("1. Run Existing Model on Sample Data")
        print("2. Create training and validation folders")
        print("3. Train a new model")
        print("4. Display Model Metrics")
        print("5. Exit")
        choice = input("\nEnter your choice (1/2/3/4/5): ")

        if choice == '1':
            print("\nID Coins...")
            run_model.id_coins(sample_dir, model_dir, train_dir)
        elif choice == '2':
            print("\nCreating training and validation folders...")
            split_data.split_data_into_train_val(main_dir, train_dir, val_dir, val_ratio=0.2)
        elif choice == '3':
            print("\nTraining a new model...")
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)
            build_model.create_model(train_dir, val_dir, model_dir)
        elif choice == '4':
            while True:
                print("\nPlease select a visualization option:")
                print("1. Bar Chart of Model Metrics")
                print("2. Box Plot of Model Metrics")
                print("3. Scatter Plot of Model Metrics")
                print("4. Return to Main Menu")
                sub_choice = input("\nEnter your choice (1/2/3/4): ")

                if sub_choice in ['1', '2', '3']:
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    data = {metric: [] for metric in metrics}
                    data['epoch'] = []
                    for model_file in model_files:
                        model_path = os.path.join(model_dir, model_file)
                        checkpoint = torch.load(model_path)
                        data['epoch'].append(checkpoint['epoch'])
                        for metric in metrics:
                            data[metric].append(checkpoint[metric])
                    df = pd.DataFrame(data)

                    if sub_choice == '1':
                        df.plot(x='epoch', y=metrics, kind='bar')
                        plt.title('Bar Chart of Model Metrics')
                    elif sub_choice == '2':
                        df[metrics].plot(kind='box')
                        plt.title('Box Plot of Model Metrics')
                    elif sub_choice == '3':
                        print("\nPlease select two metrics to plot against each other:")
                        for i, metric in enumerate(metrics, start=1):
                            print(f"{i}. {metric.capitalize()}")
                        x_metric = metrics[int(input("\nEnter the number of the first metric: ")) - 1]
                        y_metric = metrics[int(input("Enter the number of the second metric: ")) - 1]
                        plt.scatter(df[x_metric], df[y_metric])
                        plt.xlabel(x_metric.capitalize())
                        plt.ylabel(y_metric.capitalize())
                        plt.title(f'Scatter Plot of {x_metric.capitalize()} vs {y_metric.capitalize()}')
                    plt.show()
                elif sub_choice == '4':
                    break
                else:
                    print("\nInvalid choice. Please enter a number between 1 and 4.")
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")
