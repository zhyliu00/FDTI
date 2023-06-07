import torch
import os
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dnn
from dgl.utils import expand_as_pair,check_eq_shape
import dgl
class Model(nn.Module):
    def __init__(self,in_feats,hid_feat,layers):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_feats,hid_feat)
        self.conv_list = nn.ModuleList()
        self.layers = layers
        self.hid_feat = hid_feat
        self.in_feat = in_feats
        self.conv_list.append(dnn.SAGEConv(hid_feat,hid_feat,'pool'))
        for i in range(1, layers):
            self.conv_list.append(dnn.SAGEConv(hid_feat,hid_feat,'pool'))
        self.fc2 = nn.Linear(hid_feat,1)
        self.activation_fn = torch.tanh
        # self.weight = torch.nn.Parameter(torch.ones((len(self.conv_list))))
    def forward(self,blocks,x):

        h1 = x
        h1 = self.activation_fn(self.fc1(h1))
        for l, (layer, block) in enumerate(zip(self.conv_list,blocks)):
            
            out_this_layer = layer(block,h1,block.edata['w'])

            if(l != self.layers-1):
                out_this_layer = self.activation_fn(out_this_layer)

            hdst = h1[:block.number_of_dst_nodes()]
            h1 = hdst +  out_this_layer

        mask = (h1[:,] == 0).all(dim=1)
        h1 = self.activation_fn(h1)
        out_num = self.fc2(h1)
        return out_num, mask
    def inference(self,g,x,batch_size,device):
        x = self.activation_fn(self.fc1(x))
        for l, layer in enumerate(self.conv_list):
            y = torch.zeros(g.number_of_nodes(),
                            self.hid_feat)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()).to(device), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                # Copy the features of necessary input nodes to GPU
                h = x[input_nodes].to(device)

                h1 = h
                out_this_layer = layer(block,h1,block.edata['w'])
                if(l != self.layers-1):
                    out_this_layer = self.activation_fn(out_this_layer)

                hdst = h1[:block.number_of_dst_nodes()]
                h1 = hdst + out_this_layer
                y[output_nodes] = h1.cpu()
            x = y
        mask = (y[:, ] == 0).all(dim=1)
        y = self.activation_fn(y).to(device)
        out_num = self.fc2(y)
        return out_num,mask
    def save(self,path,name):
        if(not os.path.exists(path)):
            os.mkdirs(path)

