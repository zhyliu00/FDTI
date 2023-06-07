import time
import json
import dgl
import numpy as np
import torch
import networkx as nx
import argparse
from model import Model
import datetime
import os
from utils import check_path
import copy


curr_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='manhattan', type=str)
parser.add_argument('--threshold',default=5,type=int)
parser.add_argument('--test_setting',default='manhattan',type=str)
parser.add_argument('--test_folder',default='None',type=str)
parser.add_argument('--layers',default=4,type=int)
parser.add_argument('--hidden_states',default=256,type=int)

args = parser.parse_args()
hidden = args.hidden_states
threshold = args.threshold
test_setting = args.test_setting
setting = args.setting
curr_time = args.test_folder
layers = args.layers
dgl2nx = json.load(open('./data/{}_dgl2nx.json'.format(test_setting)))['test']
nx2dgl = json.load(open('./data/{}_nx2dgl.json'.format(test_setting)))['test']
config = {
    'device':'cpu',
    'batch':4096,
    'result_path' : os.path.join(curr_path , "outputs/{}/".format(setting)  + curr_time + '/results/'),  # path to save results
    'model_path' : os.path.join(curr_path , "outputs/{}/".format(setting)  + curr_time + '/models/'),  # path to save models
    'save_every' :10,
    'layers' : layers,
    'hidden_states':hidden
}
print('result_path : {}, model_path : {}'.format(config['result_path'],config['model_path']))

def calc_RMSE(y_pred,y_gnd):
    if(torch.is_tensor(y_pred[0])):
        y_pred = torch.stack(y_pred)
    if(torch.is_tensor(y_gnd[0])):
        y_gnd = torch.stack(y_gnd)
    msk = y_gnd > 5.0
    msk_large = y_gnd < 100
    msk = msk * msk_large
    y_gnd = y_gnd[msk]
    y_pred = y_pred[msk]
    return torch.sqrt(torch.mean((y_pred - y_gnd)**2))
def calc_MAPE(y_pred,y_gnd):
    if(torch.is_tensor(y_pred[0])):
        y_pred = torch.stack(y_pred)
    if(torch.is_tensor(y_gnd[0])):
        y_gnd = torch.stack(y_gnd)
    msk = y_gnd > 5.0
    msk_large = y_gnd < 100
    msk = msk * msk_large
    y_gnd = y_gnd[msk]
    y_pred = y_pred[msk]
    abs_error = torch.abs(y_pred - y_gnd)
    return torch.mean(abs_error/y_gnd)

def dgl_next_node(k):
    k = str(int(k))
    now_nx = dgl2nx[k]
    next_nx = now_nx + 1
    next_nx = str(next_nx)
    if (next_nx in nx2dgl.keys()):
        next_dgl = nx2dgl[next_nx]
    else:
        next_dgl = -1
    return next_dgl
def convert(x):
    return float(x)
def tes(outmodel,inmodel,dg,steps):

    feat = dg.ndata['f'].to(config['device'])
    volume = torch.sum(feat[:, 1:5], dim=1)
    msk = volume < 5.0
    msk_large = volume > 100.0
    msk = msk + msk_large
    msk = msk.to(config['device'])
    e_weight = dg.edata['w']
    feat = torch.cat((feat[:,0].unsqueeze(1),volume.unsqueeze(1),feat[:,5:]),dim=1)
    feat.to(config['device'])
    volume_gnd = copy.deepcopy(volume).to(config['device'])
    dg.ndata['volume_pred'] = copy.deepcopy(volume).unsqueeze(1).to(config['device'])
    outmodel.eval()
    inmodel.eval()
    results = {}
    res = {}
    # add 'gnd' attribute to each nodes
    for k in dg.nodes():
        results[int(k)] = {
            'pred': [],
            'gnd': []
        }

    dg.ndata['volume_gnd'] = copy.deepcopy(volume_gnd).to(config['device'])
    msk3 = volume_gnd < threshold
    msk += msk3
    dg.ndata['f'] = feat

    with torch.no_grad():
        start_time = time.time()
        for cur_step in range(max(steps)):
            print('now_step is {}'.format(cur_step))

            # calc 10 step, store the result in dg.ndata[volume_pred][0:10]
            feat = dg.ndata['f'].to(config['device'])
            out,msk2 = outmodel.inference(dg,feat,config['batch'],config['device'])
            end_time = time.time()
            print('inference cost {}s'.format(end_time - start_time))
            in_,msk2 = inmodel.inference(dg,feat,config['batch'],config['device'])
            out = out.squeeze(1).clamp(min=0.0)
            in_ = in_.squeeze(1).clamp(min=0.0)
            discount = pow(0.5, cur_step)
            tmp = dg.ndata['volume_pred'][:,-1] - discount *  out + discount *  in_
            tmp = tmp.clamp(min=0.0)
            res = torch.zeros_like(tmp)
            res -= 1

            
            for k, val in enumerate(tmp):
                nxt_node = dgl_next_node(k)
                if(nxt_node == -1):
                    continue
                else:
                    res[nxt_node] = val
            # print(tmp[:5],dg.ndata['volume_gnd'][:5])
            dg.ndata['volume_pred']=torch.cat((dg.ndata['volume_pred'],res.unsqueeze(1)),dim=1)
            dg.ndata['f'][:,1] = copy.deepcopy(dg.ndata['volume_pred'][:,-1]).to(config['device'])
            start_time = time.time()
    # aggregate the result
    msk_ = msk.to(config['device']) + msk2.to(config['device'])
    msk_ = ~msk_
    for k in dg.nodes():
        now = k
        step = 1
        if(msk_[k]==False):
            continue
        now = dgl_next_node(now)
        while(now!=-1 and len(dg.ndata['volume_pred'][now])>step):
            if(dg.ndata['volume_gnd'][now]==-1):
                break
            if(dg.ndata['volume_pred'][now][step] == -1):
                break
            results[int(k)]['pred'].append(dg.ndata['volume_pred'][now][step])
            results[int(k)]['gnd'].append(dg.ndata['volume_gnd'][now])
            step+=1
            now = dgl_next_node(now)

    result_steps = {}
    for step in steps:
        result_steps[step] = {
            'pred':[],
            'gnd':[]
        }
    # use results to calculate horizon result
    for k,v in results.items():
        for now_step in range(len(v['pred'])):
            for step in steps:
                if(now_step==step - 1 and step <= len(v['pred'])):
                    result_steps[step]['pred'].append(v['pred'][now_step])
                    result_steps[step]['gnd'].append(v['gnd'][now_step])
    result_metric = {

    }
    for step in steps:
        rmse = calc_RMSE(result_steps[step]['pred'],result_steps[step]['gnd'])
        mape = calc_MAPE(result_steps[step]['pred'],result_steps[step]['gnd'])
        result_metric[step] = {
            'rmse':float(rmse),
            'mape':float(mape)
        }
    print(result_metric)
    for k,v in results.items():
        v['pred'] = list(v['pred'])
        v['gnd'] = list(v['gnd'])
    json.dump(result_metric,open(config['result_path'] + '/result_{}_metric.json'.format(test_setting),'w'),indent=2)
    json.dump(results,open(config['result_path']+'/result_{}_nodes.json'.format(test_setting),'w'),default=convert,indent=2)
if __name__ == '__main__':
    outmodel = Model(6,config['hidden_states'],config['layers']).to(config['device'])
    inmodel = Model(6,config['hidden_states'],config['layers']).to(config['device'])
    state_in = torch.load(config['model_path'] + 'inmodel.pth',map_location=config['device'])
    state_out = torch.load(config['model_path'] + 'outmodel.pth',map_location=config['device'])
    inmodel.load_state_dict(state_in)
    outmodel.load_state_dict(state_out)
    phase = 'test'
    (dg_test,), _ = dgl.load_graphs('./data/{}_{}.dgl'.format(test_setting, phase))
    dg_test = dg_test.to(config['device'])
    steps = [1,3,5]
    results = tes(outmodel,inmodel,dg_test,steps)