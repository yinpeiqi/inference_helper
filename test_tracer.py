import torch.fx
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import dgl
import time
from dgl.data import CiteseerGraphDataset
from inference_helper import InferenceHelper
from inference_helper.tracer import symbolic_trace


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        graph = graph.local_var()
        if not self._allow_zero_in_degree:
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                src_prefix_shape + (self._num_heads,) + (self._out_feats,))
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]
                h_dst = h_dst[:graph.number_of_dst_nodes()]
                dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # # compute softmax
        graph.edata.update({'a': self.attn_drop(edge_softmax(graph, e))})
        # # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                            fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            resval = self.res_fc(h_dst).view(dst_prefix_shape + (-1, self._out_feats))
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.hidden_features = num_hidden
        self.heads = heads
        self.out_features = num_classes
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g[l], h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g[-1], h).mean(1)
        return logits

    def inference(self, g, batch_size, device, x):
        for l, layer in enumerate(self.gat_layers):
            if l != self.num_layers - 1:
                y = torch.zeros(g.number_of_nodes(), self.heads[l] * self.hidden_features)
            else:
                y = torch.zeros(g.number_of_nodes(), self.out_features)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=4)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)

                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l == self.num_layers - 1:
                    logits = h.mean(1)
                    y[output_nodes] = logits.cpu()
                else:
                    h = h.flatten(1)
                    y[output_nodes] = h.cpu()

            x = y

        return y


def train():
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

    model = GAT(3, in_feats, hidden_feature, num_classes, [2, 2, 2], F.relu, 0.5, 0.5, 0.5, 0.5)
    model = symbolic_trace(model)
    model.recompile()
    print(model.graph)
    print(model.code)

    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in tqdm.tqdm(range(10)):
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
        st = time.time()
        helper = InferenceHelper(model, 20, torch.device('cuda'), debug = True)
        helper_pred = helper.inference(g, feat)
        helper_score = (torch.argmax(helper_pred, dim=1) == labels).float().sum() / len(helper_pred)
        cost_time = time.time() - st
        print("Helper Inference: {}, inference time: {}".format(helper_score, cost_time))
        
        if hasattr(model, "inference"):
            st = time.time()
            pred = model.inference(g, 20, torch.device('cuda'), feat)
            func_score = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)
            cost_time = time.time() - st
            print("Origin Inference: {}, inference time: {}".format(func_score, cost_time))

if __name__ == '__main__':
    train()