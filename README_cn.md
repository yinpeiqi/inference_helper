# inference_helper
随着输入的图越来越大，现有的硬件设备已经不能满足GNN模型在图上的全邻居训练方式。越来越多的模型采用了采样mini-batch训练的方式。其中，最常用两种采样方式就是邻居采样和子图采样。然而在推理的时候，为了保证精度，推理只能使用全邻居训练，不可以采样。由于CPU的计算速度太慢，我们一般都会采用GPU进行加速推理。但是GPU的显存有限，我们不能将一整张图放进去做全图推理。这导致我们需要mini-batch全图推理。而在mini-batch的全图推理时，由于需要节点所有的邻居信息，即使batch_size很小，生成的图还是会很大。并且，中间层的特征可能会被多个不同的批次所利用，我们只能每一个批次都生成一次。这里产生了大量的重复运算。为了解决这个问题，DGL的实现方法是一层一层地进行推理，即先算完当前层的所有节点的特征，再用该特征去计算下一层。详情可参考[这里](https://docs.dgl.ai/guide/minibatch-inference.html)。

在DGL的文档中我们可以发现，手动实现一个minibatch推理很繁琐。而随着模型变得越来越复杂，用户自己去学习并使用mini-batch推理的难度也逐渐增大。如果有一个通用的推理工具将会大大减少用户的学习成本，提高用户的开发效率。本工作提出了一个根据输入的模型进行自动推理的工具inference_helper。用户只需要写模型，inference_helper会自动追踪forward函数中的消息传递层，并将forward函数切割成数个单层函数。inference_helper还提供了一个推理接口，用户仅需往接口里输入跟forward函数一样的输入，inference_helper会进行自适应推理。

## 开始
```
git clone https://github.com/yinpeiqi/inference_helper.git
cd inference_helper
python setup.py install
```

## 用户手册
原本inference的使用方式：参考dgl/examples/pytorch/graphsage/train_sampling.py
```python
pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
```
使用inference helper:
```python
from inference_helper import InferenceHelper

helper = InferenceHelper(model, args.batch_size, device)
pred = helper.inference(g, nfeat)
```