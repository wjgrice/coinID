import build_model
import split_data
import run_model


def run(main_dir, train_dir, val_dir, sample_dir, model_dir):
    while True:
        print("Please select an option:")
        print("1. Train a new model")
        print("2. Test the existing model")
        print("3. Create training and validation folders")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            print("Training a new model...")
            build_model.create_model(train_dir, val_dir)
        elif choice == '2':
            print("ID Coins...")
            run_model.id_coins(sample_dir, model_dir)
            # Call your testing function here
        elif choice == '3':
            print("Creating training and validation folders...")
            split_data.split_data_into_train_val(main_dir, train_dir, val_dir, val_ratio=0.2)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")
