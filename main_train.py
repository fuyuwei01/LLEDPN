
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import pytorch_ssim
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from LOLDataSet import LOLDataTrainSet
# Params
parser = argparse.ArgumentParser(description='PyTorch LLEDPN')
parser.add_argument('--model', default='LLEDPN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--train_data', default='./LOLdataset', type=str, help='path of train data')

parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')

args = parser.parse_args()
batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
save_dir = os.path.join('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


##模型
class DnCNN(nn.Module):
    def __init__(self, depth=4, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1,
                      bias=True))
        layers.append(nn.Mish(inplace=True))
        for _ in range(depth - 2):
            ##15层的异构卷积+Mish
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1,
                          bias=True))
            layers.append(nn.Mish(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        layers2 = []

        layers2.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers2.append(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1,
                      bias=True))
        layers2.append(nn.Mish(inplace=True))
        for _ in range(depth - 2):
            ##15层的异构卷积+Mish
            layers2.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers2.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1,
                          bias=True))
            layers2.append(nn.Mish(inplace=True))
        layers2.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn1 = nn.Sequential(*layers)
        self.dncnn2 = nn.Sequential(*layers2)
        self.Conv1 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,bias=False)
        # self._initialize_weights()

    def forward(self, x):
        out1 = self.dncnn1(x)
        out2 = self.dncnn2(x)
        cat = torch.cat((out1,out2),1)
        out = self.Conv1(cat)
        return out



##useful
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
        ##找出存档中最大的epoch
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    print(model)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
        ##读取存档
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = pytorch_ssim.SSIM(window_size = 11)
    if cuda:
        model = model.cuda()
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        # criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    for epoch in range(initial_epoch, n_epoch):

        DDataset = LOLDataTrainSet(root_dir=".\LOLdataset")
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, lowhigh in enumerate(DLoader):
            optimizer.zero_grad()
            if cuda:
                low, high = lowhigh[0].cuda(), lowhigh[1].cuda()# x low;y high

            loss = 1-criterion(model(low), high)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, 485 // batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')


        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))






