from typing import Callable, Dict, List, Union

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        rel_names: List[str],
        num_bases: int,
        norm: str = 'right',
        weight: bool = True,
        bias: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = None,
        self_loop: bool = False,
    ):
        super().__init__()
        self._rel_names = rel_names
        self._num_rels = len(rel_names)
        self._conv = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(
            in_feats, out_feats, norm=norm, weight=False, bias=False) for rel in rel_names})
        self._use_weight = weight
        self._use_basis = num_bases < self._num_rels and weight
        self._use_bias = bias
        self._activation = activation
        self._dropout = nn.Dropout(dropout) if dropout is not None else None
        self._use_self_loop = self_loop

        if weight:
            if self._use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feats, out_feats), num_bases, self._num_rels)
            else:
                self.weight = nn.Parameter(torch.Tensor(
                    self._num_rels, in_feats, out_feats))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
            nn.init.zeros_(self.bias)

        if self_loop:
            self.self_loop_weight = nn.Parameter(
                torch.Tensor(in_feats, out_feats))
            nn.init.xavier_uniform_(
                self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

    def _apply_layers(
        self,
        ntype: str,
        inputs: torch.Tensor,
        inputs_dst: torch.Tensor = None,
    ) -> torch.Tensor:
        x = inputs

        if inputs_dst is not None:
            x += torch.matmul(inputs_dst[ntype], self.self_loop_weight)

        if self._use_bias:
            x += self.bias

        if self._activation is not None:
            x = self._activation(x)

        if self._dropout is not None:
            x = self._dropout(x)

        return x

    def forward(
        self,
        hg: dgl.DGLHeteroGraph,
        x_author, x_institution, x_paper
    ) -> Dict[str, torch.Tensor]:
        hg = hg.local_var()
        inputs = {"author": x_author, "institution": x_institution, "paper": x_paper}

        if self._use_weight:
            weight = self.basis() if self._use_basis else self.weight
            weight_dict = {self._rel_names[i]: {'weight': w.squeeze(
                dim=0)} for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            weight_dict = {}

        if self._use_self_loop:
            if hg.is_block:
                inputs_dst = {ntype: h[:hg.num_dst_nodes(
                    ntype)] for ntype, h in inputs.items()}
            else:
                inputs_dst = inputs
        else:
            inputs_dst = None

        x = self._conv(hg, inputs, mod_kwargs=weight_dict)
        del inputs, weight_dict

        author_ret = self._apply_layers("author", x["author"], inputs_dst)
        inst_ret = self._apply_layers("institution", x["institution"], inputs_dst) 
        paper_ret = self._apply_layers("paper", x["paper"], inputs_dst)
        del x, inputs_dst
        return author_ret, inst_ret, paper_ret


class EntityClassify(nn.Module):
    def __init__(
        self,
        hg: dgl.DGLHeteroGraph,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_bases: int,
        num_layers: int,
        norm: str = 'right',
        layer_norm: bool = False,
        input_dropout: float = 0,
        dropout: float = 0,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
        self_loop: bool = False,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation
        self._rel_names = sorted(list(set(hg.etypes)))
        self._num_rels = len(self._rel_names)

        if num_bases < 0 or num_bases > self._num_rels:
            self._num_bases = self._num_rels
        else:
            self._num_bases = num_bases

        self._layers = nn.ModuleList()

        if num_layers == 1:
            self._layers.append(RelGraphConvLayer(
                in_feats,
                out_feats,
                self._rel_names,
                self._num_bases,
                norm=norm,
                self_loop=self_loop,
            ))

        else:
            self._layers.append(RelGraphConvLayer(
                in_feats,
                hidden_feats,
                self._rel_names,
                self._num_bases,
                norm=norm,
                self_loop=self_loop,
            ))

            for _ in range(1, num_layers - 1):
                self._layers.append(RelGraphConvLayer(
                    hidden_feats,
                    hidden_feats,
                    self._rel_names,
                    self._num_bases,
                    norm=norm,
                    self_loop=self_loop,
                ))

            self._layers.append(RelGraphConvLayer(
                hidden_feats,
                out_feats,
                self._rel_names,
                self._num_bases,
                norm=norm,
                self_loop=self_loop,
            ))

        if layer_norm:
            self._layer_norms = nn.ModuleList()

            for _ in range(num_layers - 1):
                self._layer_norms.append(nn.LayerNorm(hidden_feats))
        else:
            self._layer_norms = None

    def _apply_layers(
        self,
        layer_idx: int,
        x_author, x_institution, x_paper
    ) -> Dict[str, torch.Tensor]:

        if self._layer_norms is not None:
            x_author = self._layer_norms[layer_idx](x_author)

        if self._activation is not None:
            x_author = self._activation(x_author)

        x_author = self._dropout(x_author)

        if self._layer_norms is not None:
            x_institution = self._layer_norms[layer_idx](x_institution)

        if self._activation is not None:
            x_institution = self._activation(x_institution)

        x_institution = self._dropout(x_institution)

        if self._layer_norms is not None:
            x_paper = self._layer_norms[layer_idx](x_paper)

        if self._activation is not None:
            x_paper = self._activation(x_paper)

        x_paper = self._dropout(x_paper)
        return x_author, x_institution, x_paper

    def forward(
        self,
        hg: Union[dgl.DGLHeteroGraph, List[dgl.DGLHeteroGraph]],
        x_author, x_institution, x_paper
    ) -> Dict[str, torch.Tensor]:

        for i, layer in enumerate(self._layers):
            x_author, x_institution, x_paper = layer(hg[i], x_author, x_institution, x_paper)

            if i < self._num_layers - 1:
                x_author, x_institution, x_paper = self._apply_layers(i, x_author, x_institution, x_paper)

        return x_author, x_institution, x_paper

    def inference(
        self,
        hg: dgl.DGLHeteroGraph,
        batch_size: int,
        num_workers: int,
        embedding_layer: nn.Module,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        for i, layer in enumerate(self._layers):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                hg,
                {ntype: hg.nodes(ntype) for ntype in hg.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

            if i < self._num_layers - 1:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self._hidden_feats) for ntype in hg.ntypes}
            else:
                y = {ntype: torch.zeros(hg.num_nodes(
                    ntype), self._out_feats) for ntype in hg.ntypes}

            for in_nodes, out_nodes, blocks in dataloader:
                in_nodes = {rel: nid
                            for rel, nid in in_nodes.items()}
                out_nodes = {rel: nid
                             for rel, nid in out_nodes.items()}
                block = blocks[0].to(device)

                if i == 0:
                    h = embedding_layer(in_nodes=in_nodes, device=device)
                else:
                    h = {ntype: x[ntype][in_nodes[ntype]]
                         for ntype in hg.ntypes}

                h_list = layer(block, h['author'], h["institution"], h["paper"])
                h = {"author": h_list[0], "institution": h_list[1], "paper": h_list[2]}

                if i < self._num_layers - 1:
                    h_list = self._apply_layers(i, h['author'], h["institution"], h["paper"])
                    h = {"author": h_list[0], "institution": h_list[1], "paper": h_list[2]}

                for ntype in h:
                    if ntype in out_nodes:
                        y[ntype][out_nodes[ntype]] = h[ntype].cpu()

            x = y

        return x
