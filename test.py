import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet50, resnet34 ,resnet101
from loadc import CaptchaFolder

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # device = torch.cuda.set_device(1)
    device = torch.device('cuda')
    print("using {} device.".format(device))
    print(device)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    cudnn.deterministic = True
    cudnn.benchmark = True
    train_data = CaptchaFolder(img_path="/home/jyang/dataset-2000/pngs",
                               label_path="/home/jyang/dataset-2000/labels-2000.csv",
                               transform=data_transform["train"],
                               split='TRAIN')
    # val_data = CaptchaFolder(img_path="/home/jyang/dataset-2000/pngs",
    #                         label_path="/home/jyang/dataset-2000/labels-2000.csv",
    #                          transform=data_transform["val"],
    #                         split='VAL')
    val_data = CaptchaFolder(img_path="/home/jyang/dataset-2000/pngs",
                               label_path="/home/jyang/dataset-2000/labels-2000.csv",
                               transform=data_transform["train"],
                               split='TRAIN')
    # print(train_data[0][1][0])
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size, shuffle=False,
                                               num_workers=nw)
    train_num = len(train_data)
    val_num = len(val_data)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet50()
    net.fc = nn.Linear(net.fc.in_features,36)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    epochs = 10
    best_acc = 0.0
    save_path = './resNet50.pth'
    train_steps = 1200
    keys = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        # print("hehe")
        for step, data in enumerate(train_bar):
            images, labels = data
            # print(labels)
            labels_unique = set(labels)
            labels_onehot = torch.zeros(size=(len(labels), len(keys)))
            # print(labels_onehot)

            for idx in range(batch_size):
                for idy in range(len(keys)):
                    if labels[idx] == keys[idy]:
                        labels_onehot[idx][idy] = 1
            # print(labels_onehot)
            # print(labels_onehot)
            # label = np.zeros(batch_size,36)
            # print(label)
            # print(labels)
            # print(torch.tensor(labels))
            # print(labels)
            optimizer.zero_grad()
            logits = net(images.to(device))
            # print(logits)
            loss = loss_function(logits, labels_onehot.to(device))
            loss.backward()
            optimizer.step()
            # print("festival")
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_loss = 0
            for val_data in val_bar:
                val_images, val_labels = val_data
                labels_onehot = torch.zeros(size=(len(val_labels), len(keys)))
                for idx in range(batch_size):
                    for idy in range(len(keys)):
                        if val_labels[idx] == keys[idy]:
                            labels_onehot[idx][idy] = 1
                # print(labels_onehot)
                # print(val_labels)
                labels_t = torch.zeros(len(val_labels))
                # print(labels_t)
                for idx in range(batch_size):
                    for idy in range(len(keys)):
                        if labels_onehot[idx][idy] == 1:
                            labels_t[idx] = idy
                # print(labels_t)
                # print(labels_t)
                # print(labels_t)
                outputs = net(val_images.to(device))
                # print(outputs)
                # print(outputs)
                # print(val_labels.to(device))
                loss = loss_function(outputs,labels_onehot.to(device) )
                # print(loss)
                # print(loss)
                # print(loss)

                predict_y = torch.max(outputs, dim=1)[1]
                # print(predict_y)
                # print(val_labels)
                acc += torch.eq(predict_y, labels_t.to(device)).sum().item()
                val_loss +=loss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_loss / train_steps , val_accurate))

        filename = "log_01_c.txt"
        with open(filename, 'a') as file_object:
            file_object.write(str(epoch+1))
            file_object.write("\t")
            file_object.write(str(running_loss / train_steps))
            file_object.write("\n\t")
            file_object.write(str(val_loss / train_steps))
            file_object.write("\n")
            file_object.write("\n\t")
            file_object.write(str(val_accurate))
            file_object.write("\n")



        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)



        #train_writer = tf.summary.FileWriter(FLAGS.log_dir + 'train', sess.graph)
        #eval_writer = tf.summary.FileWriter(FLAGS.log_dir + 'eval')

        # train_summary
        #train_writer.add_summary(train_summary, global_step)
        #train_writer.flush()

        # eval_summary
        #eval_writer.add_summary(eval_summary, global_step)
        #eval_writer.flush()


    print('Finished Training')

if __name__ == '__main__':
    main()