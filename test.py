import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import time
import tqdm
from model.gcn import StochasticTwoLayerGCN
from model.sage import SAGE
from model.gat import  GAT
from dgl.data import CiteseerGraphDataset, RedditDataset
from inference_helper import InferenceHelper, EdgeControlInferenceHelper, AutoInferenceHelper

def load_reddit():
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    return g, data.num_classes

def load_ogb(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['label'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels

def train(_class):
    dataset = load_ogb("ogbn-products")
    # dataset = load_reddit()
    g : dgl.DGLHeteroGraph = dataset[0]
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    feat = g.ndata['feat']
    labels = g.ndata['label']
    num_classes = dataset[1]
    in_feats = feat.shape[1]
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    hidden_feature = 128

    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 25, 50])
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid, sampler,
        batch_size=2000,
        shuffle=True,
        drop_last=False,
        num_workers=4)

    if _class == StochasticTwoLayerGCN:
        model = StochasticTwoLayerGCN(in_feats, hidden_feature, num_classes)
    elif _class == SAGE:
        model = SAGE(in_feats, hidden_feature, num_classes, 3, F.relu, 0.5)
    elif _class == GAT:
        model = GAT(3, in_feats, hidden_feature, num_classes, [8, 8, 8], F.relu, 0.5, 0.5, 0.5, 0.5)
    else:
        raise NotImplementedError()

    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(0):
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            input_features = feat[input_nodes].to(torch.device('cuda'))
            pred = model(blocks, input_features)
            output_labels = labels[output_nodes].to(torch.device('cuda'))
            loss = loss_fcn(pred, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        print(_class.__name__)
        st = time.time()
        # helper = EdgeControlInferenceHelper(model, 2621440, torch.device('cuda'), debug = False)
        # helper = InferenceHelper(model, 2000, torch.device('cuda'), debug = False)
        helper = AutoInferenceHelper(model, torch.device('cuda'), debug = True)
        helper_pred = helper.inference(g, feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        cost_time = time.time() - st
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))

        if hasattr(model, "inference"):
            st = time.time()
            pred = model.inference(g, [100000, 50000, 50000], torch.device('cuda'), feat)
            func_score = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
            cost_time = time.time() - st
            print("Origin Inference: {}, inference time: {}".format(func_score, cost_time))


def test_GCN():
    train(StochasticTwoLayerGCN)

def test_SAGE():
    train(SAGE)

def test_GAT():
    train(GAT)

if __name__ == '__main__':
    # model = StochasticTwoLayerGCN(3, 3, 3)
    # helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    # model = SAGE(3, 3, 3, 3, F.relu, 0.5)
    # helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    # model = GAT(3, 3, 3, 3, [2, 2, 2], F.relu, 0.5, 0.5, 0.5, 0.5)
    # helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
    # test_GCN()
    # test_SAGE()
    test_GAT()
