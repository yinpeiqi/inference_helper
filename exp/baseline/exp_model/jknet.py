import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, JumpingKnowledge
import tqdm

class Concate(nn.Module):
    def forward(self, g, jumped):
        g.srcdata['h'] = jumped
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return g.dstdata['h']

class JKNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers=1,
                 mode='cat',
                 dropout=0.):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(num_layers):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        if self.mode == 'lstm':
            self.jump = JumpingKnowledge(mode, hid_dim, num_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == 'cat':
            hid_dim = hid_dim * (num_layers + 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.agge = Concate()
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, g, feats):
        feat_lst = []
        for layer in self.layers:
            feats = self.dropout(layer(g, feats))
            feat_lst.append(feats)

        jumped = self.jump(feat_lst)
        agged = self.agge(g, jumped)

        return self.output(agged)

    def inference(self, g, batch_size, device, x):
        feat_lst = []
        for l, layer in enumerate(self.layers):
            feat_lst.append(torch.zeros(g.num_nodes(), self.n_hidden))

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                
                h = layer(block, h)
                h = self.dropout(h)

                feat_lst[-1][output_nodes] = h.cpu()

            x = feat_lst[-1]

        y = torch.zeros(g.num_nodes(), self.n_classes)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)

        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            block = blocks[0]

            block = block.int().to(device)
            h_lst = []
            for feat in feat_lst:
                h_lst.append(feat[input_nodes].to(device))
            
            jumped = self.jump(feat_lst)
            agged = self.agge(g, jumped)
            output = self.output(agged)

            y[output_nodes] = output.cpu()
        return y
