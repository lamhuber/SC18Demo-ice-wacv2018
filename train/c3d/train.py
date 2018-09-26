from __future__ import print_function
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

sys.path.insert(0, '../../')
import config as cfg
from datasets import c3d_loader as loader
from datasets import CNNDataLayer as DataLayer
from models import C3D as Model

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.001)
        nn.init.constant_(m.bias.data, 0.001)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_sets = {
        phase: DataLayer(
            data_root=args.data_root,
            sessions=getattr(args, phase+'_session_set'),
            loader=loader,
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: data.DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for phase in args.phases
    }

    model = Model().apply(weights_init).to(device)
    air_criterion = nn.L1Loss().to(device)
    bed_criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs+1):
        # Learning rate scheduler
        if epoch%5 == 0 and args.lr >= 1e-05:
            args.lr = args.lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        air_errors = {phase: 0.0 for phase in args.phases}
        bed_errors = {phase: 0.0 for phase in args.phases}

        start = time.time()
        for phase in args.phases:
            training = phase=='train'
            if training:
                model.train(True)
            else:
                if epoch%args.test_interval == 0:
                    model.train(False)
                else:
                    continue

            with torch.set_grad_enabled(training):
                for batch_idx, (data, air_target, bed_target) in enumerate(data_loaders[phase]):
                    batch_size = data.shape[0]
                    data = data.to(device)
                    air_target = air_target.to(device)
                    bed_target = bed_target.to(device)

                    air_output, bed_output = model(data)
                    air_loss = air_criterion(air_output, air_target)
                    bed_loss = bed_criterion(bed_output, bed_target)
                    air_errors[phase] += air_loss.item()*batch_size
                    bed_errors[phase] += bed_loss.item()*batch_size
                    if args.debug:
                        print(air_loss.item(), bed_loss.item())

                    if training:
                        optimizer.zero_grad()
                        loss = air_loss + bed_loss
                        loss.backward()
                        optimizer.step()
        end = time.time()

        if epoch%args.test_interval == 0:
            snapshot_path = './snapshots'
            if not os.path.isdir(snapshot_path):
                os.makedirs(snapshot_path)
            snapshot_name = 'epoch-{}-air-{}-bed-{}.pth'.format(
                epoch,
                float("{:.2f}".format(air_errors['test']/len(data_loaders['test'].dataset)*412)),
                float("{:.2f}".format(bed_errors['test']/len(data_loaders['test'].dataset)*412)),
            )
            torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))

        print('Epoch {:2}, | '
              'train loss (air): {:4.2f} (bed): {:4.2f}, | '
              'test loss (air): {:4.2f} (bed): {:4.2f}, | '
              'running time: {:.2f} sec'.format(
                  epoch,
                  air_errors['train']/len(data_loaders['train'].dataset)*412,
                  bed_errors['train']/len(data_loaders['train'].dataset)*412,
                  air_errors['test']/len(data_loaders['test'].dataset)*412,
                  bed_errors['test']/len(data_loaders['test'].dataset)*412,
                  end-start,
              ))
