#!/usr/bin/env python
# coding: utf-8

import ogb
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import dgl
import torch
import numpy as np
import time
import tqdm
import dgl.function as fn
import numpy as np
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gc
from hgt_240m import HGT
from rgcn_240m import EntityClassify
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
from inference_helper import HeteroInferenceHelper

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, default='dataset/', help='Directory to download the OGB dataset.')
    parser.add_argument('--graph-path', type=str, default='dataset/mag240m_kddcup2021/dgl_graph_processed', help='Path to the graph.')
    parser.add_argument('--full-feature-path', type=str, default='dataset/full.npy',
                        help='Path to the features of all nodes.')
    parser.add_argument('--batch-size', default=500, type=int)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument("--model", default='RGCN', type=str)
    parser.add_argument('--free-rate', help="free memory rate", type=float, default=0.8)
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--hidden-feats', default=128, type=int)
    parser.add_argument('--num-layers', default=2, type=int)
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs.')
    parser.add_argument('--model-path', type=str, default='dataset/model.pt', help='Path to store the best model.')
    parser.add_argument('--submission-path', type=str, default='dataset/results', help='Submission directory.')
    args = parser.parse_args()

    t2 = time.time()
    dataset = MAG240MDataset()
    (hg,), _ = dgl.load_graphs(args.graph_path)
    hg = hg.formats(['csc'])
    t3 = time.time()
    print('Load graph:', t3-t2)

    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    # num_features = dataset.num_paper_features // 4 # TODO
    num_features = 100
    author_feats = torch.FloatTensor(np.zeros(shape=(dataset.num_authors, num_features)))
    paper_feats = torch.FloatTensor(np.zeros(shape=(dataset.num_papers, num_features)))
    institution_feats = torch.FloatTensor(np.zeros(shape=(dataset.num_institutions, num_features)))
    pin_memory_inplace(author_feats)
    pin_memory_inplace(paper_feats)
    pin_memory_inplace(institution_feats)
    print("Load features:", time.time()-t3)
    device = torch.device("cuda")
    node_dict = {}
    edge_dict = {}
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in hg.etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

    if args.model == "RGCN":
        model = EntityClassify(
            hg,
            num_features,
            args.hidden_feats,
            dataset.num_classes,
            2,
            args.num_layers,
            norm='right',
            layer_norm=False,
            input_dropout=0.1,
            dropout=0.5,
            activation=F.relu,
            self_loop=False,
        ).to(device)
    elif args.model == "HGT":
        model = HGT(hg,
            node_dict, edge_dict,
            n_inp=num_features,
            n_hid=args.hidden_feats,
            n_out=dataset.num_classes,
            n_layers=args.num_layers,
            n_heads=2,
            use_norm = True).to(device)

    with torch.no_grad():
        if args.auto:
            st = time.time()
            helper = HeteroInferenceHelper(model, torch.device(device), debug = True, free_rate=args.free_rate)
            ret = helper.inference_240m(hg, author_feats, institution_feats, paper_feats, dataset, torch.device(device))
            print("inference time:", time.time() - st)
        else:
            sst = time.time()
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
            if args.ratio:
                print("ratio:", args.ratio)
                nids = hg.nodes('paper')[:int(hg.num_nodes('paper') * args.ratio)]
            else:
                nids = hg.nodes('paper')
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {'paper': nids},
                # {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                use_uva=True,
                drop_last=False, device=device, num_workers=0)

            st = time.time()
            y = {ntype: torch.zeros(hg.num_nodes(ntype), dataset.num_classes) for ntype in hg.ntypes}

            for ntype in hg.ntypes:
                pin_memory_inplace(y[ntype])

            for step, (in_nodes, out_nodes, blocks) in enumerate(tqdm.tqdm(dataloader)):
                in_nodes = {rel: nid for rel, nid in in_nodes.items()}
                out_nodes = {rel: nid for rel, nid in out_nodes.items()}
                blocks = [block.to(device) for block in blocks]

                new_feat = gather_pinned_tensor_rows(author_feats, in_nodes['author'])
                new_feat2 = gather_pinned_tensor_rows(institution_feats, in_nodes['institution'])
                new_feat3 = gather_pinned_tensor_rows(paper_feats, in_nodes['paper'])
                author_out, inst_out, paper_out = model(blocks, new_feat, new_feat2, new_feat3)
                logits = {"author": author_out.cpu(), "institution": inst_out.cpu(), "paper": paper_out.cpu()}
                torch.cuda.synchronize()

                for ntype in logits:
                    if ntype in out_nodes:
                        y[ntype][out_nodes[ntype]] = logits[ntype].cpu()

                del new_feat, new_feat2, new_feat3
                del logits, author_out, inst_out, paper_out
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("step:", step, "; time:", time.time()-st)
            print("inference time:", time.time()-sst)
