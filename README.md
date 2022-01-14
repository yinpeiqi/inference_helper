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

## Background
## Motivation
## Design