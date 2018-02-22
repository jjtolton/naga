import collections
import copy
import functools
import types
from six import with_metaclass


def decorator(d):
    "Make function d a decorator: d wraps a function fn."

    def _d(fn):
        return functools.update_wrapper(d(fn), fn)

    return _d


decorator = decorator(decorator)


@decorator
def trace(fn):
    level = 0

    def _trace(*args, **kwargs):
        _trace.level += 1
        _level = _trace.level
        kwalargs = sfinputs(args, kwargs)
        output = dict(
            fn=fn.__name__,
            kwalargs=kwalargs,

        )
        print(' ' * _level + '-->{fn}({kwalargs})'.format(**output))
        res = fn(*args, **kwargs)
        postoutput = copy.copy(output)
        postoutput['res'] = res
        _trace.level -= 1

        print(' ' * _trace.level + '<--{fn}({kwalargs})={res}'.format(**postoutput))
        return res

    _trace.level = level
    return _trace


def sfinputs(args, kwargs):
    def sfargskwargs(_args, _kwargs):
        return '{}'.format(','.join(filter(lambda x: x, [_args, _kwargs])))

    def sfkwargs(kwargs):
        return ','.join(map(lambda x, y: '{}={}'.format(x, y), *zip(*kwargs.items()))) if kwargs else None

    def sfargs(args):
        _args = ','.join(map(str, args)) if args else None
        return _args

    _args = sfargs(args)
    _kwargs = sfkwargs(kwargs)
    kwalargs = sfargskwargs(_args, _kwargs)
    return kwalargs


def sffncall(fn, args, kwargs):
    sfin = sfinputs(args, kwargs)
    res = '{fn}({sfin})'.format(**dict(
        fn=fn.__name__,
        sfin=sfin
    ))
    return res


class NamespacedMeta(type):
    """Metaclass to convert all methods to static methods"""

    def __init__(cls, o, bases, ns):
        super(type, NamespacedMeta).__init__(o)

        for n, f in cls.__dict__.items():
            if isinstance(f, (types.MethodType, types.FunctionType)):
                setattr(cls, n, staticmethod(f))


class Namespaced(with_metaclass(NamespacedMeta)):
    """Inheriting from this will make all methods of the class static methods."""
    pass


Result = collections.namedtuple('Result', field_names=['args', 'kwargs'])

if __name__ == '__main__':
    @trace
    def recurse(n):
        return n if n < 1 else recurse(n - 1)


    @trace
    def oddrecurse(n):
        return n if n < 1 else oddrecurse(n - 1) if n % 2 == 1 else evenrecurse(n - 1)


    @trace
    def evenrecurse(n):
        return n if n < 1 else oddrecurse(n - 1) if n % 2 == 1 else evenrecurse(n - 1)


    print(oddrecurse(10))


class nil:
    def __bool__(self):
        return False

    def __repr__(self):
        return 'nil'