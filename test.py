import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model import StochasticTwoLayerGCN
from dgl.data import CiteseerGraphDataset
from inference_helper import InferenceHelper

def train():
    dataset = CiteseerGraphDataset()
    g : dgl.DGLHeteroGraph = dataset[0]
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    feat = g.ndata['feat']
    labels = g.ndata['label']
    num_class = dataset.num_classes
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
    model = StochasticTwoLayerGCN(in_feats, hidden_feature, num_class)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(20):
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            output_labels = labels[output_nodes].to(torch.device('cuda'))
            input_features = blocks[0].ndata['feat']['_N']
            pred = model(blocks, input_features)
            loss = loss_fcn(pred, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # if epoch % 10 == 0:
            #     print("Epoch {}: {}".format(epoch, (torch.argmax(pred, dim=1) == output_labels).float().sum() / len(pred)))
    
    with torch.no_grad():
        st = time.time()
        pred = model.inference(g, 20, torch.device('cuda'), feat)
        func_score = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
        cost_time = time.time() - st
        print("Origin Inference: {}, inference time: {}".format(func_score, cost_time))

        st = time.time()
        helper = InferenceHelper(model, hidden_feature, num_class)
        print(time.time()-st)
        helper_pred = helper.inference(g, 20, torch.device('cuda'), feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        cost_time = time.time() - st
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))


if __name__ == '__main__':
    train()
