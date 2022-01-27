import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from model.gcn import StochasticTwoLayerGCN
from model.sage import SAGE
from model.gat import  GAT
from dgl.data import CiteseerGraphDataset
from inference_helper import InferenceHelper


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 3

class GCN_1(GCN):
    def forward(self, graph, x0):
        x1 = F.relu(self.conv1(graph, x0))
        x2 = F.relu(self.conv2(graph, x1))
        x3 = F.relu(self.conv3(graph, x2))
        return x3 + x2 + x1

class GCN_2(GCN):
    def forward(self, blocks, x0):
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1 = F.relu(self.conv1(blocks[0], (x0, x_dst0)))
        x_dst1 = x1[:blocks[1].number_of_dst_nodes()]
        x2 = F.relu(self.conv2(blocks[1], (x1, x_dst1)))
        x_dst2 = x2[:blocks[2].number_of_dst_nodes()]
        x3 = F.relu(self.conv3(blocks[2], (x2, x_dst2)))
        return x3 + x2 + x1

class GCN_3(GCN):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__(in_features, hidden_features, out_features)
        self.conv0_1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv0_2 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv0_3 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv1_1 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv1_2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv2_1 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv2_2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 4

    def forward(self, blocks, x0):
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1_1 = F.relu(self.conv0_1(blocks[0], (x0, x_dst0)))
        x1_2 = F.relu(self.conv0_2(blocks[0], (x0, x_dst0)))
        sum_1 = x1_1 + x1_2
        sum_dst1 = sum_1[:blocks[1].number_of_dst_nodes()]
        x2_1 = F.relu(self.conv1_1(blocks[1], (sum_1, sum_dst1)))
        x_dst1_2 = x1_2[:blocks[1].number_of_dst_nodes()]
        x2_2 = F.relu(self.conv1_2(blocks[1], (x1_2, x_dst1_2)))
        sum_2 = x2_1 + x2_2
        sum_dst2 = sum_2[:blocks[2].number_of_dst_nodes()]
        x3_1 = F.relu(self.conv2_1(blocks[2], (sum_2, sum_dst2)))
        sum_2_2 = x1_2.mean() + x2_1
        sum_dst2_2 = sum_2_2[:blocks[2].number_of_dst_nodes()]
        x3_2 = F.relu(self.conv2_2(blocks[2], (sum_2_2, sum_dst2_2)))
        x1_3 = F.relu(self.conv0_3(blocks[0], (x0, x_dst0)))
        sum_4 = x3_1 + x3_2 + x1_3.mean()
        sum_dst4 = sum_4[:blocks[3].number_of_dst_nodes()]
        x_4 = F.relu(self.conv3(blocks[3], (sum_4, sum_dst4)))
        return x_4


def train(_class):
    dataset = CiteseerGraphDataset(verbose=False)
    g : dgl.DGLHeteroGraph = dataset[0]
    train_mask = g.ndata['train_mask']
    feat = g.ndata['feat']
    labels = g.ndata['label']
    num_classes = dataset.num_classes
    in_feats = feat.shape[1]
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    hidden_feature = 256

    model = _class(in_feats, hidden_feature, num_classes)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.n_layers)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid, sampler,
        batch_size=20,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            output_labels = labels[output_nodes].to(torch.device('cuda'))
            input_features = blocks[0].ndata['feat']['_N']
            pred = model(blocks, input_features)
            loss = loss_fcn(pred, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        print(_class.__name__)
        if hasattr(model, "inference"):
            st = time.time()
            pred = model.inference(g, 20, torch.device('cuda'), feat)
            func_score = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
            cost_time = time.time() - st
            print("Origin Inference: {}, inference time: {}".format(func_score, cost_time))

        st = time.time()
        helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
        helper_pred = helper.inference(g, feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        cost_time = time.time() - st
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))


def test_GCN_3():
    train(GCN_3)

if __name__ == '__main__':
    test_GCN_3()