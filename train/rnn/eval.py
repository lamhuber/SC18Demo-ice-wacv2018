from __future__ import print_function
import os
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, '../../')
import config as cfg
from datasets import RNNDataLayer as DataLayer
from models import RNN as Model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_set = DataLayer(
        data_root=args.data_root,
        sessions=args.test_session_set,
    )

    data_loader = data.DataLoader(
        data_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = Model().to(device)
    model.load_state_dict(torch.load('../../pretrained_models/rnn.pth'))
    model.train(False)
    air_criterion = nn.L1Loss().to(device)
    bed_criterion = nn.L1Loss().to(device)
    air_errors = 0.0
    bed_errors = 0.0

    start = time.time()
    with torch.set_grad_enabled(False):
        for batch_idx, (data, init, air_target, bed_target) in enumerate(data_loader):
            print('{} {:3.3f}%'.format('Processed', 100.0*batch_idx/len(data_loader)))
            batch_size = data.shape[0]
            data = data.to(device)
            init = init.to(device)
            air_target = air_target.to(device)
            bed_target = bed_target.to(device)

            air_output, bed_output = model(data, init)
            air_loss = air_criterion(air_output, air_target)
            bed_loss = bed_criterion(bed_output, bed_target)
            air_errors += air_loss.item()*batch_size
            bed_errors += bed_loss.item()*batch_size
    end = time.time()

    print('Processed all, test loss (air): {:4.2f} (bed): {:4.2f}, | '
          'running time: {:.2f} sec'.format(
              air_errors/len(data_loader.dataset)*412,
              bed_errors/len(data_loader.dataset)*412,
              end-start,
          ))
