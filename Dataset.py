import glob
import random
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):

    def __init__(self, root, transforms_=None, unaligned=False, mode='training_data'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/canon' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/iphone' % mode) + '/*.*'))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
            return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':

    transforms_ = [transforms.Resize((40,40)),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = torch.utils.data.DataLoader(ImageDataset('/home/ahmed/Research/datasets/dped/iphone/',transforms_=transforms_, unaligned=True),batch_size=2, shuffle=True)

    for ii,input, in enumerate(dataloader):
        print(input["B"].size())
        break