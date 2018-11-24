import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import torchvision
from torch.optim import Adam
# from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn
import time
from Loss.loss import CustomLoss
from data_processor.datagen import get_data_loader
from Models.model1 import PIXOR
from utils import get_model_name, load_config, plot_bev, plot_label_map
from post_process.postprocess import non_max_suppression
import sys
sys.path.insert(0, './')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# argumentparse
parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=1, help="batch size of the data")
parser.add_argument('-e', '--epochs', type=int, default=100, help='epoch of the train')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

# import visualize
# writer = SummaryWriter()

batch_size = args.batch_size
learning_rate = args.learning_rate
max_epochs = args.epochs

# best_test_loss = np.inf
use_cuda = torch.cuda.is_available()

# load dataset
config_name = 'config.json'
config, _, _, _ = load_config(config_name)
train_data_loader, val_data_loader = get_data_loader(batch_size=batch_size, use_npy=config['use_npy'], frame_range=config['frame_range'])

# model
if use_cuda:
    device = 'cuda'
    cudnn.benchmark = True
    net = PIXOR(config['use_bn']).to(device)
else:
    device = 'cpu'
    net = PIXOR(config['use_bn']).to(device)

# loss
criterion = CustomLoss(device=device, num_classes=1)

# create your optimizer
# optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
optimizer = Adam(net.parameters())


def train(epoch):
    net.train()          # tran mode
    total_loss = 0.

    for batch_idx, (pc_feature, label_map) in enumerate(train_data_loader):
        N = pc_feature.size(0)
        pc_feature = pc_feature.to(device)
        label_map = label_map.to(device)

        # Forward
        pc_feature = Variable(pc_feature)
        label_map = Variable(label_map)
        predictions = net(pc_feature)
        loss = criterion(predictions, label_map)
        loss /= N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.data[0])


        if (batch_idx) % 20 == 0:
            print('train epoch [%d/%d], iter[%d/%d], lr %.7f, aver_loss %.5f' % (epoch,
                                                                                 max_epochs, batch_idx,
                                                                                 len(train_data_loader), learning_rate,
                                                                                 total_loss / (batch_idx + 1)))

        # # visiualize scalar
        # if epoch % 10 == 0:
        #     label_img = tools.labelToimg(labels[0])
        #     net_out = out[0].data.max(1)[1].squeeze_(0)
        #     out_img = tools.labelToimg(net_out)
        #     writer.add_scalar("loss", loss, epoch)
        #     writer.add_scalar("total_loss", total_loss, epoch)
        #     writer.add_scalars('loss/scalar_group', {"loss": epoch * loss,
        #                                              "total_loss": epoch * total_loss})
        #     writer.add_image('Image', imgs[0], epoch)
        #     writer.add_image('label', label_img, epoch)
        #     writer.add_image("out", out_img, epoch)

        assert total_loss is not np.nan
        assert total_loss is not np.inf

    # model save
    if not os.path.exists('pretrained_models'):
        os.makedirs('pretrained_models')
    if (epoch) % 2 == 0:
        torch.save(net.state_dict(), 'pretrained_models/model_%d.pth'%epoch)  # save for 5 epochs
    total_loss /= len(train_data_loader)
    print('train epoch [%d/%d] average_loss %.5f' % (epoch, max_epochs, total_loss))


def val(epoch):
    net.eval()
    total_loss = 0.
    for batch_idx, (pc_feature, labels) in enumerate(val_data_loader):
        N = pc_feature.size(0)
        if use_cuda:
            pc_feature = pc_feature.cuda()
            labels = labels.cuda()
        pc_feature = Variable(pc_feature)    # , volatile=True
        labels = Variable(labels)  # , volatile=True

        out = net(pc_feature)
        loss = criterion(out, labels)
        loss /= N
        total_loss += loss.data[0]

        if (batch_idx + 1) % 10 == 0:
            print('test epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch,
                                                                       max_epochs, batch_idx, len(val_data_loader),
                                                                       total_loss / (batch_idx + 1)))



    total_loss /= len(val_data_loader)
    print('val epoch [%d/%d] average_loss %.5f' % (epoch, max_epochs, total_loss))

    global best_test_loss 
    best_test_loss = np.inf
    if best_test_loss > total_loss:
        best_test_loss = total_loss
        print('best loss....')


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # if not os.path.exists('pretrained_model/'):
    #         os.makedirs('pretrained_model/')
    for epoch in range(max_epochs):
        train(epoch)
        # val(epoch)
        # adjust learning rate
        if epoch  == 2:
            learning_rate *= 0.1
            optimizer.param_groups[0]['lr'] = learning_rate
