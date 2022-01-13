from operator import getitem
from torch.fx import Graph

from .constants import OUTPUT, PLACEHOLDER, CALL_FUNCTION
from .utils import arg_trace


class GraphRewriter():
    @staticmethod
    def blocks_to_graph(graph: Graph):
        blocks = None
        for node in graph.nodes:
            if node.op == PLACEHOLDER and blocks is None:
                blocks = node
            elif node.op == CALL_FUNCTION and node.target == getitem and node.args[0].name == blocks.name:
                node.replace_all_uses_with(blocks)
                graph.erase_node(node)
        graph.lint()

    @staticmethod
    def remove_unused_nodes(graph: Graph):
        used_set = set()
        for node in graph.nodes.__reversed__():
            if node.op != OUTPUT and node.name not in used_set:
                graph.erase_node(node)
            else:
                used_args = arg_trace(node.args)
                for arg in used_args:
                    used_set.add(arg)
        graph.lint()
