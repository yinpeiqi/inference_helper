from .utils import arg_trace


class NodeRelation:
    def __init__(self, name, lineno):
        self.name = name
        self.lineno = lineno
        self.use = []
        self.be_used = []

    def set_use(self, use):
        self.use.append(use)

    def set_be_used(self, be_used):
        self.be_used.append(be_used)

    def __str__(self):
        return "{}: {}; {}".format(self.lineno, self.use, self.be_used)

    @staticmethod
    def get_node_relation(node_list):
        name2lineno_map = {}
        node_relation = [None for _ in range(len(node_list))]
        for lineno, node in enumerate(node_list):            
            name2lineno_map[node.name] = lineno
            node_relation[lineno] = NodeRelation(node.name, lineno)
            used_args = arg_trace(node.args)
            used_linenos = [name2lineno_map[arg] for arg in used_args]
            for used_lineno in used_linenos:
                node_relation[used_lineno].set_be_used(lineno)
                node_relation[lineno].set_use(used_lineno)
        return node_relation
    
    @staticmethod
    def print_node_relation(node_list):
        name2lineno_map = {}
        node_relation = [None for _ in range(len(node_list))]
        for lineno, node in enumerate(node_list):            
            if "getitem" not in node.name and "dst_nodes" not in node.name and "blocks" not in node.name:
                if "conv" in node.name:
                    print("usecase \"\\n{}\\n\" as {}".format(node.name, node.name))
                else:
                    print("usecase \"{}\" as {}".format(node.name, node.name))
            name2lineno_map[node.name] = lineno
            node_relation[lineno] = NodeRelation(node.name, lineno)
            used_args = arg_trace(node.args)
            used_linenos = [name2lineno_map[arg] for arg in used_args]
            for used_lineno in used_linenos:
                if "getitem" not in node.name and "dst_nodes" not in node.name and "blocks" not in node.name:
                    if node_relation[used_lineno].name != "blocks" and "getitem" not in node_relation[used_lineno].name and "dst_nodes" not in node_relation[used_lineno].name:
                        print("{} --> {}".format(node_relation[used_lineno].name, node.name))
