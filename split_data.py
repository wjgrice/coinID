import os
import shutil
from sklearn.model_selection import train_test_split


def split_data_into_train_val(main_dir, train_dir, val_dir, val_ratio=0.2):
    # Clear the train_dir and val_dir
    for directory in [train_dir, val_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    # Get the list of class names
    classes = os.listdir(main_dir)

    # For each class...
    for class_name in classes:
        # Get a list of all the image filenames
        image_filenames = os.listdir(os.path.join(main_dir, class_name))

        # Split the filenames into train and validation sets
        train_filenames, val_filenames = train_test_split(image_filenames, test_size=val_ratio)

        # Copy the train images into the train directory
        for filename in train_filenames:
            src_path = os.path.join(main_dir, class_name, filename)
            dst_path = os.path.join(train_dir, class_name, filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        # Copy the validation images into the validation directory
        for filename in val_filenames:
            src_path = os.path.join(main_dir, class_name, filename)
            dst_path = os.path.join(val_dir, class_name, filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copyfile(src_path, dst_path)
