import torch
import torch.nn as nn


class SliencedModule(nn.Module):
    def forward(self, x):
        return x

class ModuleSliencer():
    def __init__(self, module):
        self.module = module
        self.slienced = []

    def slience(self, slience_modules):
        for k in self.module._modules.keys():
            v = self.module._modules[k]
            for clazz in slience_modules:
                if isinstance(v, clazz):
                    self.slienced.append((self.module._modules, k, v))
                    self.module._modules[k] = SliencedModule()

    def unslience(self):
        for (obj, k, v) in self.slienced:
            obj[k] = v
