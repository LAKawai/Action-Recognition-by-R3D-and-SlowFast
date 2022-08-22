import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import config
from network import R3D, SlowFast
from dataloaders.dataset import VideoDataset

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
resume_epoch = 0
use_test = True
test_interval = 1
snapshot = 1
lr = 1e-2
dataset = 'ucf101'
num_classes = 0

if dataset == 'ucf101':
    num_classes = 101

save_dir_root = config.Path.save_dir_root()

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(config.Path.save_dir_root(), 'run_' + str(run_id))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model_name = 'SlowFast'
# model_name = 'R3D'
save_name = model_name + '-' + dataset


if __name__ == '__main__':
    if model_name == 'R3D':
        model = R3D.R3DModel(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif model_name == 'SlowFast':
        model = SlowFast.resnet50(class_num=num_classes)
        train_params = model.parameters()
    else:
        print('not proper model')
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # summary(model, torch.rand(1, 3, 64, 224, 224).shape)
    if resume_epoch == 0:
        print('training {} from beginning...'.format(model_name))
    else:
        state_dict_path = os.path.join(save_dir, save_name + '_epoch_' + str(resume_epoch - 1) + '.pth.tar')
        checkpoint = torch.load(state_dict_path, map_location=lambda storage, loc: storage)
        print('initializing weights from: {}...'.format(state_dict_path))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    print('total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print('training model on {} dataset...'.format(dataset))
    train_data = VideoDataset(dataset=dataset, split='train', clip_len=16)
    valid_data = VideoDataset(dataset=dataset, split='valid', clip_len=16)
    test_data = VideoDataset(dataset=dataset, split='test', clip_len=16)

    train_loader = DataLoader(train_data, batch_size=24, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=8, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=8, num_workers=4)

    train_valid_loaders = {'train': train_loader, 'valid': valid_loader}
    train_valid_sizes = {x: len(train_valid_loaders[x].dataset) for x in ['train', 'valid']}
    test_size = len(test_loader.dataset)

    for epoch in range(resume_epoch, epochs):
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()
            _loss, _correct = 0., 0.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for _images, _labels in train_valid_loaders[phase]:
                images = _images.to(device)
                labels = _labels.to(device)
                optimizer.zero_grad()
                if phase == 'train':
                    outputs = model.forward(images)
                else:
                    with torch.no_grad():
                        outputs = model.forward(images)

                pred = torch.max(outputs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _loss += loss.item() * images.size(0)
                _correct += torch.sum(pred == labels)

            epoch_loss = _loss / train_valid_sizes[phase]
            epoch_acc = _correct / train_valid_sizes[phase]

            if phase == 'train':
                scheduler.step()

            stop_time = timeit.default_timer()
            print('epoch: {}, phase: {}, loss: {}, acc: {}'.format(epoch, phase, epoch_loss, epoch_acc))
            print('execution time: {}'.format(stop_time - start_time))

        if (epoch + 1) % snapshot == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
                       os.path.join(save_dir, save_name + '_epoch_' + str(epoch) + '.pth.tar'))
            print('save model {}\n'.format(os.path.join(save_dir, save_name + '_epoch_' + str(epoch) + '.pth.tar')))

        if use_test and (epoch + 1) % test_interval == 0:
            model.eval()
            start_time = timeit.default_timer()
            loss_test, correct_test = 0., 0.
            for _images, _labels in test_loader:
                images = _images.to(device)
                labels = _labels.to(device)

                with torch.no_grad():
                    outputs = model.forward(images)

                pred = torch.max(outputs, 1)[1]
                loss = criterion(outputs, labels)

                loss_test += loss.item() * images.size(0)
                correct_test += torch.sum(pred == labels)

            epoch_loss = loss_test / test_size
            epoch_acc = correct_test / test_size

            print('**********')
            print('test  epoch: {}, loss: {}, acc: {}'.format(epoch, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print('execution time: {}'.format(stop_time - start_time))
