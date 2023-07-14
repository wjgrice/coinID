import torch as torch

import console_frontend

main_dir = 'C:\\Users\\wjgri\\PycharmProjects\\coinID\\data\\coins'
train_dir = 'C:\\Users\\wjgri\\PycharmProjects\\coinID\\data\\train'
val_dir = 'C:\\Users\\wjgri\\PycharmProjects\\coinID\\data\\val'
sample_dir = 'C:\\Users\\wjgri\\PycharmProjects\\coinID\\data\\test_sample'
model_dir = 'C:\\Users\\wjgri\\PycharmProjects\\coinID\\models'

if __name__ == "__main__":
    print(torch.cuda.is_available())

    console_frontend.run(main_dir, train_dir, val_dir, sample_dir, model_dir)
