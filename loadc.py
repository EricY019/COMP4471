import os
import re
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def MakeDataset(root, file, split):
    img_name = []
    labels = []
    num_train = 1200
    num_val = 400
    num_test = 400

    with open(file, 'rb') as text:
        reader = text.read().split()
        for i in range(len(reader)):
            labels.append(re.findall(',([0-9A-Z]*)', str(reader[i], encoding="utf-8"))[0])
    # print(labels)
    if (split == 'TRAIN'):
        video_root = os.path.join(root, 'train')
        labels = labels[0:num_train]
        for v in range(1, num_train + 1):
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v), '_', str(frame)]))
                temp = temp + '.png'
                img_name.append(temp)
    elif (split == 'VAL'):
        video_root = os.path.join(root, 'validation')
        labels = labels[num_train:num_train + num_val]
        for v in range(1, num_val + 1):
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v+num_train), '_', str(frame)]))
                temp = temp + '.png'
                # print(temp)
                img_name.append(temp)
    elif (split == 'TEST'):
        video_root = os.path.join(root, 'test')
        for v in range(1, num_test + 1):
            labels = labels[num_train + num_val:num_train + num_val + num_test]
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v+num_train+num_val), '_', str(frame)]))
                temp = temp + '.png'
                img_name.append(temp)
    return img_name, labels[0]


class CaptchaFolder(data.Dataset):  # split: one of 'TRAIN', 'VAL', 'TEST'
    def __init__(self, img_path, label_path, transform=None, batch_size=4, split=None):
        self.img_path = img_path
        self.label_file = label_path
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.batch_size = batch_size
        self.transform = transform
        self.batch_size = batch_size
        self.imgs, self.labels = MakeDataset(img_path, label_path, split)

    def __getitem__(self, index):
        img_pth = self.imgs[index % len(self.imgs)]
        label = self.labels[index % len(self.labels)]
        img = Image.open(img_pth).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

train_data = CaptchaFolder(img_path="/home/jyang/dataset-2000/pngs",
                            label_path="/home/jyang/dataset-2000/labels-2000.csv",
                            split='TRAIN')
print()