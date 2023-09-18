

class NetworkNode:
    def __init__(self):
        self.input_nodes = []
        self.output_nodes = []

    def connect_to(self, node):
        node.input_nodes.append(self)
        self.output_nodes.append(node)

    def forward_graph(self, inputs, **kwargs):
        node_outputs = self.forward(inputs, **kwargs)
        if self.output_nodes:
            outputs = []
            for output_node in self.output_nodes:
                outputs.extend(output_node.forward_graph(node_outputs, **kwargs))
            return outputs
        else:
            return node_outputs
