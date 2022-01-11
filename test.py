import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
from model.gcn import StochasticTwoLayerGCN
from model.sage import SAGE
from model.gat import  GAT
from dgl.data import CiteseerGraphDataset
from inference_helper import InferenceHelper
from memory_profiler import profile

def train(_class):
    dataset = CiteseerGraphDataset(verbose=False)
    g : dgl.DGLHeteroGraph = dataset[0]
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    feat = g.ndata['feat']
    labels = g.ndata['label']
    num_classes = dataset.num_classes
    in_feats = feat.shape[1]
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    hidden_feature = 256

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid, sampler,
        batch_size=20,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    if _class == StochasticTwoLayerGCN:
        model = StochasticTwoLayerGCN(in_feats, hidden_feature, num_classes)
    elif _class == SAGE:
        model = SAGE(in_feats, hidden_feature, num_classes, 3, F.relu, 0.5)
    elif _class == GAT:
        model = GAT(3, in_feats, hidden_feature, num_classes, [2, 2, 2], F.relu, 0.5, 0.5, 0.5, 0.5)
    else:
        raise NotImplementedError()

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
        helper = InferenceHelper(model, 20, torch.device('cuda'), debug = False)
        helper_pred = helper.inference(g, feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        cost_time = time.time() - st
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))


def test_GCN():
    train(StochasticTwoLayerGCN)

def test_SAGE():
    train(SAGE)

def test_GAT():
    train(GAT)

if __name__ == '__main__':
    model = StochasticTwoLayerGCN(3, 3, 3)
    helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    model = SAGE(3, 3, 3, 3, F.relu, 0.5)
    helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    model = GAT(3, 3, 3, 3, [2, 2, 2], F.relu, 0.5, 0.5, 0.5, 0.5)
    helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    test_GCN()
    test_SAGE()
    test_GAT()
