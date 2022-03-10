import torch.fx
import torch as th
import torch.nn as nn

import dgl
from dgl.nn import GATConv
import tqdm


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
                y = th.zeros(g.number_of_nodes(), self.heads[l] * self.hidden_features)
            else:
                y = th.zeros(g.number_of_nodes(), self.out_features)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, th.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=False,
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
