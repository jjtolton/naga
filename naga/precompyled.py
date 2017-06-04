from collections import namedtuple
from functools import wraps

from naga.tools import apply

FuncWrap = namedtuple('FuncWrap', field_names=['fn', 'args', 'kwargs'])


def iconize(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        return FuncWrap(fn, args, kwargs)

    return inner


def runall(pc_fn):
    if isinstance(pc_fn, FuncWrap):
        return pc_fn.fn(*map(lambda x: runall(x), pc_fn.args),
                        **{runall(k): runall(v) for k, v in pc_fn.kwargs.items()})
    return pc_fn


def main():
    @iconize
    def foo(x, y=None):
        return x + 2 + (y or 0)

    @iconize
    def bar(x):
        return x + 4

    @iconize
    def baz(x):
        return runall(foo(x)) + runall(bar(x))

    print(runall(foo(baz(2), y=bar(2))))


if __name__ == '__main__':
    main()
