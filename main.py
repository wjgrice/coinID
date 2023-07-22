import console_frontend
import os

# Get the absolute path to the directory the script is in
script_dir = os.path.dirname(os.path.realpath(__file__))

# Points to the parent directory, as if navigating one directory up
parent_dir = os.path.normpath(os.path.join(script_dir, '..'))

# Define paths relative to the parent directory
main_dir = os.path.join(parent_dir,'coinID', 'data', 'coins')
train_dir = os.path.join(parent_dir,'coinID', 'data', 'train')
val_dir = os.path.join(parent_dir,'coinID', 'data', 'val')
sample_dir = os.path.join(parent_dir,'coinID', 'test_sample')
model_dir = os.path.join(parent_dir,'coinID', 'data', 'models')


if __name__ == "__main__":
    console_frontend.run(main_dir, train_dir, val_dir, sample_dir, model_dir)
