import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from dataset import ImageFolder
from model import ColorNet
import matplotlib.pyplot as plt
from util import *
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

def plot_hist(tr_hist, va_hist):
    x = np.arange(len(tr_hist))
    plt.figure()
    plt.plot(x, tr_hist)
    plt.plot(x, va_hist)
    plt.legend(['Training', 'Validation'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def save_label(label, img_gray, path):
    output_ab = idx2ab(label)
    output_image = combine(img_gray, output_ab)
    plt.imsave(path, output_image)


def generateFake(test_loader, model, device, folder='output_train'):
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for i, (img_gray, labels) in enumerate(test_loader):
            img_gray = img_gray.to(device)
            labels = labels.to(device)
            output = model(img_gray)[0].cpu().numpy()
            y_pred = np.argmax(output, 0).astype('uint8')
            save_label(y_pred, img_gray, './{}/y{}.png'.format(folder, i))


def train(train_loader, model, criterion, optimizer, device, epoch):
    start = time.time()
    running_loss = 0.0
    cnt = 0
    model = model.train()
    for img_gray, labels in tqdm(train_loader):
        img_gray = img_gray.to(device)
        labels = labels.to(device)

        output = model(img_gray)
        output = output.squeeze(1)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        cnt += 1
    end = time.time()
    running_loss /= cnt
    print('\n [epoch %d] loss: %.3f elapsed time %.3f' %
            (epoch, running_loss, end-start))
    return running_loss

def validate(val_loader, model, criterion, device):
    losses = 0.0
    cnt = 0
    with torch.no_grad():
        model = model.eval()
        for img_gray, labels in tqdm(val_loader):
            #img_gray = img_gray.unsqueeze(1).float()
            img_gray = img_gray.to(device)
            labels = labels.to(device)
            output = model(img_gray)
            output = output.squeeze(1)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print('\n',losses / cnt)
    return (losses/cnt)

tr_transform = transforms.Compose([
    transforms.RandomRotation(30),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

va_transform = None


def main(args):
    N_CLASS = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ImageFolder(flag='train', data_range=(0,800), onehot=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = ImageFolder(flag='train', data_range=(800,835), onehot=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    test_data = ImageFolder(flag='test', data_range=(0,5), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1)

    #model = ECCVGenerator().to(device)
    model = ColorNet().to(device)

    print('Your network:')
    #summary(model, (1,128,128))

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trn_hist, val_hist = [], []
    for epoch in range(args.num_epochs):
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        trn_loss = train(train_loader, model, criterion, optimizer, device, epoch+1)
        print('Validation loss: ')
        val_loss = validate(val_loader, model, criterion, device)
        trn_hist.append(trn_loss)
        val_hist.append(val_loss)
        generateFake(test_loader, model, device, folder='output_test')

    plot_hist(trn_hist, val_hist)
    print('\nFinished Training, Testing on test set')
    #validate(test_loader, model, criterion, device)
    print('\nGenerating Unlabeled Result')
    
    generateFake(test_loader, model, device, folder='output_test')
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    #parser.add_argument('--num_workers', type=int, default=3)
    args = parser.parse_args()
    main(args)