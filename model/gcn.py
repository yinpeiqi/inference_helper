import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 3

    def forward(self, blocks, x0):
        # a bit modify here, for topo test
        x_dst0 = x0[:blocks[0].number_of_dst_nodes()]
        x1 = F.relu(self.conv1(blocks[0], (x0, x_dst0)))
        x_dst1 = x1[:blocks[1].number_of_dst_nodes()]
        x2 = F.relu(self.conv2(blocks[1], (x1, x_dst1)))
        x_dst2 = x2[:blocks[2].number_of_dst_nodes()]
        x3 = F.relu(self.conv3(blocks[2], (x2, x_dst2)))
        return x3

    def inference(self, g, batch_size, device, x):
        """
        Offline inference with this module
        """
        # Compute representations layer by layer
        for l, layer in enumerate([self.conv1, self.conv2, self.conv3]):
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
                num_workers=4)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
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
