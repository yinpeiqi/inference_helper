import dgl
import torch
import tqdm
from dgl.data import CiteseerGraphDataset, RedditDataset
from test import CustomDataset

if __name__ == "__main__":
    dataset = RedditDataset(self_loop=True)
    # dataset = load_ogb("ogbn-products")
    g : dgl.DGLHeteroGraph = dataset[0]
    train_mask = g.ndata['train_mask']
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]

    result = [[], [], [], [], []]
    # ori_batch_size = [100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

    b1 = [i*20 for i in range(1, 50, 1)]
    shuffle = False
    custom = True
    print(shuffle)
    for num_batch in tqdm.tqdm(b1):
        batch_size = 153500 // num_batch
        mem = 70000000 // num_batch
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        if custom:
            # print(mem)
            custom_dataset = CustomDataset(mem, g, train_nid)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, custom_dataset, sampler,
                shuffle=shuffle,
                drop_last=False,
                num_workers=0)
        else:
            dataloader = dgl.dataloading.NodeDataLoader(
                g, train_nid, sampler,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                num_workers=0)

        tot_i = 0
        max_i, max_o, max_e = 0, 0, 0
        num_batch = 0
        for input_nodes, output_nodes, blocks in dataloader:
            num_batch += 1
            tot_i += input_nodes.shape[0]
            max_o = max(max_o, output_nodes.shape[0])
            max_i = max(max_i, input_nodes.shape[0])
            max_e = max(max_e, blocks[0].num_edges())
            # print(blocks[0])
        # print(num_batch, max_i, max_o, max_e)
        # print()
        result[0].append(num_batch)
        result[1].append(max_i)
        result[2].append(max_o)
        result[3].append(max_e)
        result[4].append(tot_i)
    print(result)
