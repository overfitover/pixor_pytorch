import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import torchvision
from torch.optim import Adam
from tensorboardX import SummaryWriter
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
writer = SummaryWriter()

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
        loss, cls_loss, loc_loss = criterion(predictions, label_map)
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
    # visiualize scalar
    if True:
        if epoch % 1 == 0:
            # print ("begin summary.....")
            if True:
                b,w,h,c = pc_feature.size()
                pc_feature1 = pc_feature[0,:,:,:3].view(w,h,3).permute(2, 0, 1)
                pc_feature2 = pc_feature[0,:,:,3:6].view(w,h,3).permute(2, 0, 1)
                pc_feature3 = pc_feature[0,:,:,6:9].view(w,h,3).permute(2, 0, 1)
                pc_feature4 = pc_feature[0,:,:,9:12].view(w,h,3).permute(2, 0, 1)
                pc_feature5 = pc_feature[0,:,:,12:15].view(w,h,3).permute(2, 0, 1)
                pc_feature6 = pc_feature[0,:,:,15:18].view(w,h,3).permute(2, 0, 1)
                writer.add_image("feature1", pc_feature1, epoch)
                writer.add_image("feature2", pc_feature2, epoch)
                writer.add_image("feature3", pc_feature3, epoch)
                writer.add_image("feature4", pc_feature4, epoch)
                writer.add_image("feature5", pc_feature5, epoch)
                writer.add_image("feature6", pc_feature6, epoch)
                
            if True:
                writer.add_scalar("total_loss", total_loss, epoch)
                writer.add_scalar("cls_loss", cls_loss, epoch)
                writer.add_scalar("loc_loss", loc_loss, epoch)

                b,w,h,c = predictions.size()
                predictions0 = predictions[0,:,:,0].view(w,h,1).permute(2,0,1)
                predictions1 = predictions[0,:,:,1].view(w,h,1).permute(2,0,1)
                predictions2 = predictions[0,:,:,2].view(w,h,1).permute(2,0,1)
                predictions3 = predictions[0,:,:,3].view(w,h,1).permute(2,0,1)
                predictions4 = predictions[0,:,:,4].view(w,h,1).permute(2,0,1)
                predictions5 = predictions[0,:,:,5].view(w,h,1).permute(2,0,1)
                predictions6 = predictions[0,:,:,6].view(w,h,1).permute(2,0,1)

                writer.add_image('predictions0', predictions0, epoch)
                writer.add_image("predictions1", predictions1, epoch)
                writer.add_image("predictions2", predictions2, epoch)
                writer.add_image("predictions3", predictions3, epoch)
                writer.add_image("predictions4", predictions4, epoch)
                writer.add_image("predictions5", predictions5, epoch)
                writer.add_image("predictions6", predictions6, epoch)

            if True:
                # writer for predict 3 layer
                b,w,h,c = predictions.size()
                predictions0_3 = predictions[0,:,:,:3].view(w,h,3).permute(2,0,1)
                predictions3_6 = predictions[0,:,:,3:6].view(w,h,3).permute(2,0,1)

                writer.add_image('predict0-3', predictions0_3, epoch)
                writer.add_image('predict3-6', predictions3_6, epoch)

            if True:
                    # writer for label
                    b,w,h,c = label_map.size()
                    label_map0 = label_map[0,:,:,0].view(w,h,1).permute(2,0,1)
                    label_map1 = label_map[0,:,:,1].view(w,h,1).permute(2,0,1)
                    label_map2 = label_map[0,:,:,2].view(w,h,1).permute(2,0,1)
                    label_map3 = label_map[0,:,:,3].view(w,h,1).permute(2,0,1)
                    label_map4 = label_map[0,:,:,4].view(w,h,1).permute(2,0,1)
                    label_map5 = label_map[0,:,:,5].view(w,h,1).permute(2,0,1)
                    label_map6 = label_map[0,:,:,6].view(w,h,1).permute(2,0,1)

                    label_map0_3 = label_map[0,:,:,:3].view(w,h,3).permute(2,0,1)
                    label_map3_6 = label_map[0,:,:,3:6].view(w,h,3).permute(2,0,1)

                    writer.add_image('label_map0', label_map0, epoch)
                    writer.add_image("label_map1", label_map1, epoch)
                    writer.add_image("label_map2", label_map2, epoch)
                    writer.add_image("label_map3", label_map3, epoch)
                    writer.add_image("label_map4", label_map4, epoch)
                    writer.add_image("label_map5", label_map5, epoch)
                    writer.add_image("label_map6", label_map6, epoch)
                    writer.add_image("label_map0_3", label_map0_3, epoch)
                    writer.add_image("label_map3_6", label_map3_6, epoch)
                    

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
