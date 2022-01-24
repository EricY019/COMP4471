import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import re
from PIL import Image
import tqdm



def MakeDataset(root, file, split):
    img_name = []
    labels = []
    num_train = 1200    # Edit
    num_val = 400       # Edit
    num_test = 400      # Edit

    with open(file, 'rb') as text:
        reader = text.read().split()
        for i in range(len(reader)):
            labels.append(re.findall(',([0-9A-Z]*)', str(reader[i], encoding="utf-8"))[0])
    # print(labels)
    if (split == 'TRAIN'):
        video_root = root# os.path.join(root, 'train')
        labels = labels[0:num_train]
        for v in range(1, num_train + 1):
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v), '_', str(frame)]))
                temp = temp + '.png'
                img_name.append(temp)
    elif (split == 'VAL'):
        video_root = root# os.path.join(root, 'validation')
        labels = labels[num_train:num_train + num_val]
        for v in range(1, num_val + 1):
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v+num_train), '_', str(frame)]))
                temp = temp + '.png'
                # print(temp)
                img_name.append(temp)
    elif (split == 'TEST'):
        video_root = root# os.path.join(root, 'test')
        for v in range(1, num_test + 1):
            labels = labels[num_train + num_val:num_train + num_val + num_test]
            for frame in range(1, 26):
                temp = os.path.join(video_root, ''.join([str(v+num_train+num_val), '_', str(frame)]))
                temp = temp + '.png'
                img_name.append(temp)
    return img_name, labels

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, img_path, label_path, transform=None, batch_size=4, split=None):
        self.img_path = img_path
        self.label_file = label_path
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.batch_size = batch_size
        self.transform = transform
        self.batch_size = batch_size
        self.imgs, self.label= MakeDataset(img_path, label_path, split)
        self.caption = torch.zeros(size=(len(self.label),6))
        keys = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '<start>', '<end>')
        print(len(self.label))
        print(len(self.imgs))
        for idx in range(len(self.label)):
            self.caption[idx][0] = 36
            self.caption[idx][5] = 37
        # if split == 'TRAIN':
        for idx in range(len(self.label)):
            for idy in range(4):
                    # print(self.label[idx][idy])
                for idz in range(len(keys)):
                    print(len(self.label[idx]))
                    if self.label[idx][idy] == keys[idz]:
                        self.caption[idx][idy+1] = float(idz)

        # print(self.caption)

    def __getitem__(self, index):
        img_pth = self.imgs[index % len(self.imgs)]
        if index % len(self.imgs) % 25 >= 14:
            frame = index % len(self.imgs) - 10
        else:
            frame = index % len(self.imgs) + 10
        img_pth_2 = self.imgs[frame]
        # print("first",img_pth)
        # print("second",img_pth_2)
        label = self.caption[index % len(self.caption)]
        label = torch.tensor(label)
        # caption =[]
        # for idx in range(4):
        #     caption.append(label[idx])
        img = Image.open(img_pth).convert('RGB')
        img_2 = Image.open(img_pth_2).convert('RGB')
        captionlength = 6
        all_caption = self.caption[index % len(self.caption)].int()

        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform(img_2)
        if self.split == 'VAL':
            # print("here!")
            return img, label, captionlength, all_caption
        return img, img_2, label, captionlength

    def __len__(self):
        return len(self.imgs)


# train_data = CaptionDataset(img_path="../dataset-2000/pngs",
#                              label_path="../dataset-2000/labels-2000.csv",
#                              transform=None,
#                              split='TRAIN')
# training_data = DataLoader(dataset=CaptionDataset)
# print(training_data)

# for k in range(1000):
#     sample = training_data[k]
# print(train_data.__getitem__(index=1))