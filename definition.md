
# 问题定义
In order to support GNN full neighbour sampling inference, we need to compute the features layer by layer. So first, we will split the forward function to several convolution functions. 

给定一个有向无环图$G$，图中的节点$v_i$分为普通节点$v_{Ci}$和消息传递节点$v_{MPi}$。每个节点$v_i$和每条边$e_i$都有两个属性，一个是颜色$c_{i}$，另一个是权重$p_i$。初始状态下所有的节点以及边的的颜色都为白色。

我们可以对$G$做出以下操作：

1. 将一个白色的点染成黑色。
2. 将一条起点是黑色点的边染成黑色。

限制：使得任意两个$v_{MP}$之间的路径，一定包含黑色边。

目标：使所有黑色点和黑色边的权重$p$加和最小。


详细说明：将白色的点染成黑色表示将该点作为checkpoint，他的权重代表将数据从GPU转移到CPU的开销。黑色的边表示该数据传输从checkpoint传过去，权重代表将数据从CPU转移到GPU的开销。黑色的点的可以既有白色的边也有黑色的边，是因为我们在决定将该数据作为checkpoint时，不代表所有需要该数据的点都从checkpoint读出，也可以从GPU读出。我们的限制是，使得任意两个$v_{MP}$之间的路径包含黑色边，意思是两个消息传递module之间，一定要经过一次GPU-CPU-GPU的过程。我们的目标是使得所有黑点黑边权重加和最小，意思是所有的数据转移量最小。
