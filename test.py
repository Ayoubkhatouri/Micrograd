from graphviz import Digraph
import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        # grad is representing the derivitive of the dL/da ore dL/db ...
        self.grad = 0
        # function to calculate grad of it previous items automatically
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # this actually if we want to do smth like this a+1
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad  # local_derivative*global_derivative
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out._backward = _backward

        return out

    # this is used if we wanna do 2*a , pythhon will swap it autmatically in this case
    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):  # self/other
        return self * other**-1

    def __neg__(self):  # -self
        return self*-1

    def __sub__(self, other):  # self-other
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1-t**2)*out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward

        return out

    # lets use topological sort algorithm that build a graph from left to right and then we will reversed to call _backward
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a*b
e.label = 'e'
d = e+c
d.label = "d"
f = Value(-2.0, label="f")
L = d*f
L.label = 'L'

##########################


def trace(root):
    # build a set of all nodes and edges
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={
                  'rankdir': 'LR'})  # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph , create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (
            n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation , create an op for it
            dot.node(name=uid+n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        # connect in the top op of n2
        dot.edge(str(id(n1)), str(id(n2))+n2._op)

    return dot


L.grad = 1.0  # cause dL/dL=1
f.grad = 4.0  # cause dL/df=d=4
d.grad = -2  # cause dL/dd=f=-2
c.grad = -2  # cause dL/dc=(dL/dd)*(dd/dc)=dL/dd
e.grad = -2
a.grad = 6  # cause dL/da=(dL/de)*(de/da)
b.grad = -4

draw_dot(L).render('myGraphe', view=True)

#####################


def lol():
    h = 0.0001

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a*b
    e.label = 'e'
    d = e+c
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d*f
    L.label = 'L'
    L1 = L.data

    a = Value(2.0+h, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a*b
    e.label = 'e'
    d = e+c
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d*f
    L.label = 'L'
    L2 = L.data

    print((L2-L1)/h)


lol()

# input x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(-1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1
x1w1.label = 'x1*w1'
x2w2 = x2*w2
x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1+x2w2
x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2+b
n.label = 'n'
e = (2*n).exp()
o = (e-1)/(e+1)  # o=tan(n)
o.label = 'o'

o.backward()


draw_dot(o).render('myGraphe2', view=True)
