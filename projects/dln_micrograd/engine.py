
from dln.operator import instantiate_model

LLM = instantiate_model("gpt-35-turbo")


class Value:
    """ stores a single text value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = ""
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def match(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if self.data == other.data:
            out = Value("yes", (self, other), 'match')
        else:
            out = Value("no", (self, other), 'match')

        def _backward():
            if self.data == other.data:
                # out.grad += f"OUTPUT: {self.data}\nTARGET: {other.data}\nFeedback: OUTPUT is already similar to TARGET.\n\n"
                out.grad += f"The following text \"{self.data}\" is already similart to \"{other.data}\"\n"
            else:
                # out.grad += f"OUTPUT: {self.data}\nTARGET: {other.data}\nFeedback: OUTPUT should be more similar to TARGET.\n\n"
                out.grad += f"The following text \"{self.data}\" should be changed to be more similar to \"{other.data}\"\n"

            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def backward(self, grad=""):

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

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = grad
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={repr(self.data)}, grad={repr(self.grad)})"



class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = ""

    def parameters(self):
        return []


class Prompt(Module):

    def __init__(self, text):
        self.prompt = Value(text)

    def __call__(self, x):
        x = x if isinstance(x, Value) else Value(x)
        out = Value(LLM(self.prompt.data + x.data)[0], (self.prompt, x), 'LLM')

        def _backward():
            # grad = LLM(f"Instruction: ???\nInput: {x.data}\nOutput: {out.grad}\n\nWhat would be the instruction to go from Input to Output.\nInstruction:")[0]
            grad = LLM(f"Given the following:\n'''\nPROMPT: {self.prompt.data}\nINPUT:{x.data}\n{out.grad}\n'''\nwhat should be changed in PROMPT?")[0]
            # grad_prompt = f"{out.grad}\nGiven that  {self.prompt.data}\nINPUT:{x.data}\n{out.grad}\n'''\nwhat should be changed in PROMPT?"

            self.prompt.grad += grad
            #x.grad += out.grad

        out._backward = _backward

        return out

    def parameters(self):
        return [self.prompt]

    def __repr__(self):
        return f"Prompt({repr(self.prompt)})"


from wandb.integration.openai import autolog

autolog({"project": "lautograd", "name": "nano_dln"})

model = Prompt("Reply in French.\n")
input1 = Value("Hi, how are you?")
input2 = Value("In which country is Berlin?")
print(f"model: {model}")
print(f"input1: {input1}")
output1 = model(input1)
print(f"output1: {output1}")

output2 = model(input2)
print(f"output2: {output2}")


# Compute loss.
target1 = "Muy bien."
target2 = "alemania."

loss = output1.match(target1)
#output.grad = target
loss.backward()

new_prompt = LLM(f"Given the following:\n'''\nPROMPT: {model.prompt.data}\nFeedback: {model.prompt.grad}\n'''\nwhat should be a better PROMPT?\nPROMPT:")[0]
print(new_prompt)



from ipdb import set_trace; set_trace()

# print(f"target: {target}")
# grad_prompt, grad_input = output.backward(target)
# print(f"grad_prompt: {grad_prompt}")
# print(f"grad_input: {grad_input}")
