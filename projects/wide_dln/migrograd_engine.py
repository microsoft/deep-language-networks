from dln.operator import LLMRegistry

REGISTRY = LLMRegistry()
LLM = REGISTRY.register("gpt-4", max_tokens=512)
# LLM = REGISTRY.register("text-davinci-003", max_tokens=512)


class Value:
    """Stores a single text value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = ""
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def forward(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(LLM(f"{self.data}\n{other.data}")[0], (self, other), 'forward')
        def _backward():
            grad = LLM(
                f"Given the instruction:\n{self.data}\n\n"
                f"Given the following feedback:\n{out.grad}\n\n"
                f"What should be changed in the instruction?"
            )[0]
            self.grad = f"{self.grad}\n{grad}".strip()
        out._backward = _backward
        return out

    def match_loss(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            f"Mistake: {self.data}\nCorrect answer: {other.data}"
            if self.data != other.data else "",
            (self, other),
            'match_loss',
        )
        def _backward():
            # this just forwards the grad, but one could ask the LLM a grad for this specific mistake
            if out.grad not in self.grad:
                self.grad = f"{self.grad}\n{out.grad}".strip()
            if out.grad not in other.grad:
                other.grad = f"{other.grad}\n{out.grad}".strip()
        out._backward = _backward
        return out

    def concat_loss(self, others):
        losses = tuple([self] + others)
        for other in others:
            other = other if isinstance(other, Value) else Value(other)
        out = Value(f"{self.data}\n{other.data}".strip(), losses, 'concat')
        def _backward():
            mistakes_and_answers = "\n".join([l.data for l in losses])
            grad = LLM(
                f"Given the following pairs Mistake and Correct answer:\n"
                f"{mistakes_and_answers}\n"
                f"Provide a reason why the mistakes were made:"
            )[0]
            self.grad = f"{self.grad}\n{grad}".strip()
            for other in others:
                other.grad = f"{other.grad}\n{grad}".strip()
        out._backward = _backward
        return out

    def apply_grad(self):
        self.data = LLM(
            f"Given the instruction:\n{self.data}\n\n"
            f"Given the following feedbacks:\n{self.grad}\n\n"
            f"A better instruction would be:"
        )[0]

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={repr(self.data)}, grad={repr(self.grad)}, op={repr(self._op)})"


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = ""

    def parameters(self):
        return []


class LanguageLayer(Module):

    def __init__(self, text):
        self.prompt = Value(text)

    def __call__(self, x):
        out = self.prompt.forward(x)
        return out

    def parameters(self):
        return [self.prompt]

    def __repr__(self):
        return f"LanguageLayer({repr(self.prompt)})"


class DLN(Module):
    pass


def dfs(root):
    # just for debugging
    visited = set()
    def inner_dfs(node):
        print(node)
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                inner_dfs(child)
    inner_dfs(root)


model = LanguageLayer("Reply to the question in French.")
print(f"Model: {model}")

inputs = ["Hi, how are you doing?", "How old are you?"]  #, "In which country is Berlin?"]
true_outputs = ["Hola! Estoy bien, gracias.", "Tengo 30 años"]  #, "Alemania."]

outputs = [model(i) for i in inputs]

for i in range(2):
    losses = [o.match_loss(y) for o, y in zip(outputs, true_outputs)]
    loss = losses[0].concat_loss(losses[1:])

    print("\nForward pass:")
    dfs(loss)

    model.zero_grad()
    loss.backward()

    print("\nBackward pass:")
    dfs(loss)

    for p in model.parameters():
        p.apply_grad()

    print("\nAfter applying gradients:")
    for p in model.parameters():
        print(p)

    outputs = [model(i) for i in inputs]
    print("\nNew outputs:")
    for o in outputs:
        print(o)
