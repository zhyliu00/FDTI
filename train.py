import subprocess
import json
import time
import dgl
import numpy as np
import torch
import networkx as nx
import argparse
from model import Model
import datetime
import os
from utils import check_path
curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
# print(os.path.abspath(__file__))
curr_path = os.path.dirname(os.path.abspath(__file__))
# curr_path = os.path.abspath(__file__)
print("cur_path is : {}".format(curr_path))
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--setting', default='manhattan', type=str)
parser.add_argument('--layers', default=4, type=int)
parser.add_argument('--hidden_states', default=256, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--batch', default=512, type=int)
args = parser.parse_args()
setting = args.setting
hidden = args.hidden_states
layers = args.layers
epochs = args.epochs
batch = args.batch
config = {
    'device': torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"),
    # 'device':'cpu',
    'batch': batch,
    'lr': 5e-4,
    'max_epohs': epochs,
    'save_every': 10,
    'layers': layers,
    'hidden_states': hidden
}
print(config['device'])
config['result_path'] = os.path.join(
    curr_path, "outputs/{}/".format(setting) + curr_time + '/results/')  # path to save results
config['model_path'] = os.path.join(
    curr_path, "outputs/{}/".format(setting) + curr_time + '/models/')  # path to save models

print('result_path : {}, model_path : {}'.format(
    config['result_path'], config['model_path']))


def get_dataloader(dg_dict):
    dataloader_dict = {}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config['layers'])
    for k, dg in dg_dict.items():
        dataloader = dgl.dataloading.NodeDataLoader(
            dg, list(range(0, dg.num_nodes())), sampler,
            batch_size=config['batch'],
            shuffle=True,
            drop_last=False,
            device=config['device']
        )
        dataloader_dict[k] = dataloader

    return dataloader_dict


def train(outmodel, inmodel, dg_dict, dataloader_dict):
    opt_out = torch.optim.Adam(outmodel.parameters(), lr=config['lr'])
    opt_in = torch.optim.Adam(inmodel.parameters(), lr=config['lr'])
    # loss_fn = torch.nn.HuberLoss(reduction='sum',delta=3.0)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_arr = {
        'out': {
            'train': [],
            'val': []
        },
        'in': {
            'train': [],
            'val': []
        },
        'total_node': {
            'train': len(dg_dict['train'].ndata['f']),
            'val': len(dg_dict['val'].ndata['f'])
        }
    }
    epoch_times = []
    for epoch in range(config['max_epohs']):
        val_loss_in = 0.0
        val_loss_out = 0.0
        train_loss_in = 0.0
        train_loss_out = 0.0
        sm_train = 0
        sm_val = 0
        start_time = time.time()
        for phase in phases:
            dg_loader = dataloader_dict[phase]
            dg = dg_dict[phase]
            for step, (input_nodes, output_nodes, blocks) in enumerate(dg_loader):
                feat = dg.ndata['f'][input_nodes]
                volume = torch.sum(feat[:, 1:5], dim=1)
                feat_out = dg.ndata['f'][output_nodes]
                volume_out = torch.sum(feat_out[:, 1:5], dim=1)
                msk = volume_out < 5.0
                msk_large = volume_out > 100.0
                msk = msk + msk_large
                out_label = dg.ndata['out'][output_nodes]
                in_label = dg.ndata['in'][output_nodes]
                feat = torch.cat((feat[:, 0].unsqueeze(
                    1), volume.unsqueeze(1), feat[:, 5:]), dim=1)
                if(phase == 'val'):
                    outmodel.eval()
                    inmodel.eval()
                    with torch.no_grad():
                        out_pred, msk2 = outmodel(blocks, feat)
                        in_pred, msk2 = inmodel(blocks, feat)
                        msk = msk + msk2
                        msk = ~msk
                        out_pred = out_pred[msk]
                        in_pred = in_pred[msk]
                        out_label = out_label[msk].unsqueeze(1)
                        in_label = in_label[msk].unsqueeze(1)
                        loss_out = loss_fn(out_pred, out_label)
                        loss_in = loss_fn(in_pred, in_label)
                        sm_val += len(out_label)
                        val_loss_in += loss_in.item()
                        val_loss_out += loss_out.item()
                else:
                    outmodel.train()
                    inmodel.train()
                    out_pred, msk2 = outmodel(blocks, feat)
                    in_pred, msk2 = inmodel(blocks, feat)
                    msk = msk + msk2
                    msk = ~msk
                    out_pred = out_pred[msk]
                    in_pred = in_pred[msk]
                    out_label = out_label[msk].unsqueeze(1)
                    in_label = in_label[msk].unsqueeze(1)
                    loss_out = loss_fn(out_pred, out_label)
                    loss_in = loss_fn(in_pred, in_label)
                    sm_train += len(out_label)

                    train_loss_out += loss_out.item()
                    train_loss_in += loss_in.item()

                    opt_in.zero_grad()
                    opt_out.zero_grad()
                    loss_out.backward()
                    loss_in.backward()
                    opt_out.step()
                    opt_in.step()
        train_loss_out /= sm_train
        val_loss_out /= sm_val
        train_loss_in /= sm_train
        val_loss_in /= sm_val

        if((epoch + 1) % config['save_every'] == 0):
            torch.save(outmodel.state_dict(),
                       config['model_path'] + 'outmodel.pth')
            torch.save(inmodel.state_dict(),
                       config['model_path'] + 'inmodel.pth')
            json.dump(loss_arr, open(
                config['result_path'] + 'loss.json', 'w'), indent=2)

        loss_arr['out']['train'].append(train_loss_out)
        loss_arr['out']['val'].append(val_loss_out)
        loss_arr['in']['train'].append(train_loss_in)
        loss_arr['in']['val'].append(val_loss_in)
        end_time = time.time()

        cost_time = end_time - start_time
        epoch_times.append(cost_time)
        print('in epoch {}/{}, cost time :{}, out train loss {}, val loss {}; in train loss {}, val loss {}'.format(
            epoch, config['max_epohs'], cost_time, train_loss_out, val_loss_out, train_loss_in, val_loss_in))

    loss_arr['avg_epoch_time'] = np.mean(epoch_times)
    return loss_arr


if __name__ == "__main__":
    phases = ['train', 'val', 'test']
    dg_dict = {}
    outmodel = Model(6, config['hidden_states'],
                     config['layers']).to(config['device'])
    inmodel = Model(6, config['hidden_states'],
                    config['layers']).to(config['device'])

    for phase in phases:
        (dg_dict[phase],), _ = dgl.load_graphs(
            './data/{}_{}.dgl'.format(setting, phase))
        dg_dict[phase] = dg_dict[phase].to(config['device'])
    check_path(config['result_path'])
    check_path(config['model_path'])
    dataloader_dict = get_dataloader(dg_dict)
    loss_arr = train(outmodel, inmodel, dg_dict, dataloader_dict)

    json.dump(loss_arr, open(
        config['result_path'] + 'loss.json', 'w'), indent=2)
    curr_time = curr_time
    cmd = 'python test.py --setting {} --test_setting {} --test_folder {}  --layers {}  --hidden_states {}'.format(setting,
                                                                                                                   setting, curr_time, config['layers'], config['hidden_states'])
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
