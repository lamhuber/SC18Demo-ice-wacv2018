from __future__ import print_function
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

sys.path.insert(0, '../../')
import config as cfg
from datasets import c3d_loader as loader
from datasets import CNNDataLayer as DataLayer
from models import C3D as Model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = cfg.parse_args(parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_sets = {
        phase: DataLayer(
            data_root=args.data_root,
            sessions=getattr(args, phase+'_session_set'),
            loader=loader,
            training=False,
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: data.DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        for phase in args.phases
    }

    model = Model().to(device)
    model.load_state_dict(torch.load('../../pretrained_models/c3d.pth'))
    model.train(False)

    with torch.set_grad_enabled(False):
        for phase in args.phases:
            for batch_idx, (data, air_target, bed_target, save_path) in enumerate(data_loaders[phase]):
                print('{} {:3.3f}%'.format(phase, 100.0*batch_idx/len(data_loaders[phase])))
                batch_size = data.shape[0]
                data = data.to(device)
                air_feature, bed_feature = model.features(data)
                air_feature = air_feature.to('cpu').numpy()
                bed_feature = bed_feature.to('cpu').numpy()
                for bs in range(batch_size):
                    if not osp.isdir(osp.dirname(save_path[bs])):
                        os.makedirs(osp.dirname(save_path[bs]))
                    np.save(
                        save_path[bs],
                        np.concatenate((air_feature[bs], bed_feature[bs]), axis=0)
                    )
