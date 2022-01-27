from .utils import arg_trace
from .constants import CALL_MODULE, PLACEHOLDER, CALL_METHOD


class SplitPlanGenerator():
    def __init__(self, node_list):
        self.node_list = node_list
        self.plan = self.generate_split_linenos()

    def get_split_plan(self):
        return self.plan

    def get_node_relation(self):
        name2lineno_map = {}
        node_relation = [None for _ in range(len(self.node_list))]
        for lineno, node in enumerate(self.node_list):
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
                node_relation[used_lineno].set_be_used(lineno)
                node_relation[lineno].set_use(used_lineno)
        return node_relation

    def generate_split_linenos(self):
        blocks_name = None
        layer_nodes = []
        node_status = [[NodeStatus(0, 0, [])]]
        can_append = True
        for lineno, node in enumerate(self.node_list):
            if len(layer_nodes) > 0 and can_append:
                layer_nodes[-1].append(lineno)
            if node.op == CALL_METHOD and blocks_name in arg_trace(node.args):
                can_append = False
            elif node.op == CALL_MODULE and blocks_name in arg_trace(node.args):
                layer_nodes.append([])
                can_append = True
            elif node.op == PLACEHOLDER:
                node_status[0][0].lineno = lineno + 1
                if blocks_name is None:
                    blocks_name = node.name
        if len(layer_nodes) <= 1:
            return []
        layer_nodes[-1] = [len(self.node_list) - 1]

        node_relation = self.get_node_relation()
        for layer_id, layer in enumerate(layer_nodes):
            node_status.append([])
            for node_lineno in layer:
                curr_status = NodeStatus(node_lineno, 10000, [])
                for last_layer_status in node_status[layer_id]:
                    self.update_status(node_relation, last_layer_status, curr_status)
                node_status[-1].append(curr_status)
        return node_status[-1][0].plan[:-1]

    def update_status(self, node_relation, last_status, curr_status):
        require_input = set()
        require_output = set()
        for lineno in range(last_status.lineno, curr_status.lineno):
            if node_relation[lineno].be_used >= curr_status.lineno:
                require_output.add(lineno)
            for arg in node_relation[lineno].use:
                if arg < last_status.lineno:
                    require_input.add(arg)
        # cost calculation
        next_cost = last_status.cost + len(require_input) + len(require_output)
        if next_cost < curr_status.cost:
            curr_status.cost = next_cost
            curr_status.plan = last_status.plan.copy()
            curr_status.plan.append(curr_status.lineno)

class NodeStatus:
    def __init__(self, lineno, cost, plan):
        self.lineno = lineno
        self.plan = plan
        self.cost = cost

    def __str__(self):
        return "{}: {}; {}".format(self.lineno, self.plan, self.cost)

class NodeRelation:
    def __init__(self, name, lineno):
        self.name = name
        self.lineno = lineno
        self.use = []
        self.be_used = -1

    def set_use(self, use):
        self.use.append(use)

    def set_be_used(self, be_used):
        self.be_used = max(self.be_used, be_used)

    def __str__(self):
        return "{}: {}; {}".format(self.lineno, self.use, self.be_used)
