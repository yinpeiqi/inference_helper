import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, n_layer, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.convs = nn.ModuleList()
        self.n_layers = n_layer
        self.convs.append(dgl.nn.GraphConv(in_features, hidden_features))
        for l in range(n_layer - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_features, hidden_features))
        self.convs.append(dgl.nn.GraphConv(hidden_features, out_features))

    def forward(self, blocks, x):
        for conv in self.convs:
            x_dst = x[:blocks[0].number_of_dst_nodes()]
            x = F.relu(conv(blocks[0], (x, x_dst)))
        return x

    def inference(self, g, batch_size, device, x):
        """
        Offline inference with this module
        """
        # Compute representations layer by layer
        for l, layer in enumerate(self.convs):
            y = torch.zeros(g.number_of_nodes(),
                            self.hidden_features
                            if l != self.n_layers - 1
                            else self.out_features)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)

                # Copy the features of necessary input nodes to GPU
                h = x[input_nodes].to(device)
                # Compute output.  Note that this computation is the same
                # but only for a single layer.
                h_dst = h[:block.number_of_dst_nodes()]
                h = F.relu(layer(block, (h, h_dst)))
                # Copy to output back to CPU.
                y[output_nodes] = h.cpu()

            x = y

        return y

