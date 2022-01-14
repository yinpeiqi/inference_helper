# inference_helper

Inference Helper provides a general inference tool for minibatch training of graph neural networks on large graphs. Users only need to write a model. Inference Helper will analysis the forward function using torch FX then split it to several layers. When user call the `inference` function of inference helper, it will make use of the splitted functions for the inference step.

## Get Started
```
python setup.py install
```

## User Guide
Define a Module in Pytorch.
```python
import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(blocks[l], h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    model = SAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)

```
Use the inference helper to achieve minibatch inference layer by layer.
```python
from inference_helper import InferenceHelper

helper = InferenceHelper(model, batch_size, device, num_workers)
pred = helper.inference(g, nfeat)
```

## Motivation
If we have a massive graph with millions or even billions of nodes or edges, usually full-graph training would not work. Consider an 𝐿-layer graph convolutional network with hidden state size 𝐻 running on an 𝑁-node graph. Storing the intermediate hidden states requires 𝑂(𝑁𝐿𝐻) memory, easily exceeding one GPU’s capacity with large 𝑁. Sampling like subgraph sampling and stochastic minibatch sampling stretegy is widely adopt in this scenario.

By using the sampling strategy, user only need to write the module code just like what to do with the full graph training. However, the situation is different when inferencing. 

The inference algorithm is different from the training algorithm, as the representations of all nodes should be computed layer by layer, starting from the first layer. Specifically, for a particular layer, we need to compute the output representations of all nodes from this GNN layer in minibatches. The consequence is that the inference algorithm will have an outer loop iterating over the layers, and an inner loop iterating over the minibatches of nodes. In contrast, the training algorithm has an outer loop iterating over the minibatches of nodes, and an inner loop iterating over the layers for both neighborhood sampling and message passing. The code is shown below.

```python
class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
        self.n_layers = 2

    def forward(self, blocks, x):
        x_dst = x[:blocks[0].number_of_dst_nodes()]
        x = F.relu(self.conv1(blocks[0], (x, x_dst)))
        x_dst = x[:blocks[1].number_of_dst_nodes()]
        x = F.relu(self.conv2(blocks[1], (x, x_dst)))
        return x

    def inference(self, g, x, batch_size, device):
        """
        Offline inference with this module
        """
        # Compute representations layer by layer
        for l, layer in enumerate([self.conv1, self.conv2]):
            y = torch.zeros(g.number_of_nodes(),
                            self.hidden_features
                            if l != self.n_layers - 1
                            else self.out_features)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.number_of_nodes()), sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False)

            # Within a layer, iterate over nodes in batches
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

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
```
The inference code is quite different from the training one, which increase the overhead for user to implement a GNN model for large graphs. Inference helper can reduce this overhead, provide ready-made function interface for the inference step.
## Design
Inference Helper contains two components: function generator and inference.

Function generator aims at generate several convolution functions by spliting the forward function by each layers. Function generator use torch FX, which can provide tranformation between a pytorch module and a python computation graph IR. Function generator first trace the forward function with torch FX, and determine whether the forward function using a blocks or a graph as input. If a blocks is used, then transfrom it to graph. 
