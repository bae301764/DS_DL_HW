import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
# import some packages you need here

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.preprocess = T.Compose([T.ToTensor(),
                                    T.Normalize((0.1307), (0.3081))])
        self.file_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        data_num = self.file_list[idx]
        data = os.path.join(self.data_dir, data_num)

        img = Image.open(data)
        img = self.preprocess(img)
        label = int(data.split('_')[1][0])
        
        return img, label

# if __name__ == '__main__':

#     # write test codes to verify your implementations


