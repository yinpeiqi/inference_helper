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

class RGAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_etypes, num_layers, num_heads, dropout, pred_ntype):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        
        self.convs.append(nn.ModuleList([
            dglnn.GATConv(in_channels, hidden_channels // num_heads, num_heads, allow_zero_in_degree=True)
            for _ in range(num_etypes)
        ]))
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(nn.ModuleList([
                dglnn.GATConv(hidden_channels, hidden_channels // num_heads, num_heads, allow_zero_in_degree=True)
                for _ in range(num_etypes)
            ]))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes
        
    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            x_dst = x[:mfg.num_dst_nodes()]
            n_src = mfg.num_src_nodes()
            n_dst = mfg.num_dst_nodes()
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(mfg.edata['etype'] == j, preserve_nodes=True)
                x_skip += self.convs[i][j](subg, (x, x_dst)).view(-1, self.hidden_channels)
            x = self.norms[i](x_skip)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)


class ExternalNodeCollator(dgl.dataloading.NodeCollator):
    def __init__(self, g, idx, sampler, offset, feats, label):
        super().__init__(g, idx, sampler)
        self.offset = offset
        self.feats = feats
        self.label = label

    def collate(self, items):
        input_nodes, output_nodes, mfgs = super().collate(items)
        # Copy input features
        mfgs[0].srcdata['x'] = torch.FloatTensor(self.feats[input_nodes])
        mfgs[-1].dstdata['y'] = torch.LongTensor(self.label[output_nodes - self.offset])
        return input_nodes, output_nodes, mfgs

def train(args, dataset, g, feats, paper_offset):
    print('Loading masks and labels')
    train_idx = torch.LongTensor(dataset.get_idx_split('train')) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split('valid')) + paper_offset
    label = dataset.paper_label

    print('Initializing dataloader...')
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 25])
    train_collator = ExternalNodeCollator(g, train_idx, sampler, paper_offset, feats, label)
    valid_collator = ExternalNodeCollator(g, valid_idx, sampler, paper_offset, feats, label)
    train_dataloader = torch.utils.data.DataLoader(
        train_collator.dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        collate_fn=train_collator.collate,
        num_workers=4
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2
    )

    print('Initializing model...')
    model = RGAT(dataset.num_paper_features, dataset.num_classes, 1024, 5, 2, 4, 0.5, 'paper').cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.25)

    best_acc = 0

    for _ in range(args.epochs):
        model.train()
        with tqdm.tqdm(train_dataloader) as tq:
            for i, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                mfgs = [g.to('cuda') for g in mfgs]
                x = mfgs[0].srcdata['x']
                y = mfgs[-1].dstdata['y']
                y_hat = model(mfgs, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                acc = (y_hat.argmax(1) == y).float().mean()
                tq.set_postfix({'loss': '%.4f' % loss.item(), 'acc': '%.4f' % acc.item()}, refresh=False)

        model.eval()
        correct = total = 0
        for i, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(valid_dataloader)):
            with torch.no_grad():
                mfgs = [g.to('cuda') for g in mfgs]
                x = mfgs[0].srcdata['x']
                y = mfgs[-1].dstdata['y']
                y_hat = model(mfgs, x)
                correct += (y_hat.argmax(1) == y).sum().item()
                total += y_hat.shape[0]
        acc = correct / total
        print('Validation accuracy:', acc)

        sched.step()

        if best_acc < acc:
            best_acc = acc
            print('Updating best model...')
            torch.save(model.state_dict(), args.model_path)

def test(args, dataset, g, feats, paper_offset):
    print('Loading masks and labels...')
    valid_idx = torch.LongTensor(dataset.get_idx_split('valid')) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split('test')) + paper_offset
    label = dataset.paper_label

    print('Initializing data loader...')
    sampler = dgl.dataloading.MultiLayerNeighborSampler([160, 160])
    valid_collator = ExternalNodeCollator(g, valid_idx, sampler, paper_offset, feats, label)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2
    )
    test_collator = ExternalNodeCollator(g, test_idx, sampler, paper_offset, feats, label)
    test_dataloader = torch.utils.data.DataLoader(
        test_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator.collate,
        num_workers=4
    )

    print('Loading model...')
    model = RGAT(dataset.num_paper_features, dataset.num_classes, 1024, 5, 2, 4, 0.5, 'paper').cuda()
    model.load_state_dict(torch.load(args.model_path))

    model.eval()
    correct = total = 0
    for i, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(valid_dataloader)):
        with torch.no_grad():
            mfgs = [g.to('cuda') for g in mfgs]
            x = mfgs[0].srcdata['x']
            y = mfgs[-1].dstdata['y']
            y_hat = model(mfgs, x)
            correct += (y_hat.argmax(1) == y).sum().item()
            total += y_hat.shape[0]
    acc = correct / total
    print('Validation accuracy:', acc)
    evaluator = MAG240MEvaluator()
    y_preds = []
    for i, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(test_dataloader)):
        with torch.no_grad():
            mfgs = [g.to('cuda') for g in mfgs]
            x = mfgs[0].srcdata['x']
            y = mfgs[-1].dstdata['y']
            y_hat = model(mfgs, x)
            y_preds.append(y_hat.argmax(1).cpu())
    evaluator.save_test_submission({'y_pred': torch.cat(y_preds)}, args.submission_path)

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
    # import pdb; pdb.set_trace()
    # feats = np.memmap(args.full_feature_path, mode='r', dtype='float16', shape=(num_nodes, num_features))
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
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {'paper': hg.nodes('paper')},
                # {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                use_uva=True,
                drop_last=False, device=device, num_workers=0)

            st = time.time()
            y = {ntype: torch.zeros(hg.num_nodes(ntype), dataset.num_classes) for ntype in hg.ntypes}
            for step, (in_nodes, out_nodes, blocks) in enumerate(tqdm.tqdm(dataloader)):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
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

                del new_feat, new_feat2, new_feat3, blocks
                for ntype in logits:
                    if ntype in out_nodes:
                        del out_nodes[ntype]
                    if ntype in in_nodes:
                        del in_nodes[ntype]
                del logits["author"], logits["institution"], logits["paper"]
                del logits, author_out, inst_out, paper_out
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(torch.cuda.memory_summary())
                print("step:", step, "; time:", time.time()-st)
