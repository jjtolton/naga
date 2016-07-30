import itertools
from functools import reduce


def rreduce(fn, seq, default=None):
    """'readable reduce' - More readable version of reduce with arrity-based dispatch; passes keyword arguments
    to functools.reduce"""

    # if two arguments
    if default is None:
        return reduce(fn, seq)

    # if three arguments
    return reduce(fn, seq, default)


def get_in(d, ks, not_found=None):
    return reduce(lambda x, y: x.get(y, {}), ks, d) or not_found


def apply(fn, x):
    return fn(*x)


def first(iterable):
    return next(iter(iterable))


def last(iterable):
    a, b = itertools.tee(iterable)
    iter_len = reduce(lambda n, x: n + 1, enumerate(b), 0)
    return next(itertools.islice(a, iter_len - 1, iter_len))


def rest(iterable):
    return itertools.islice(iterable, 1, None)


def explode(*ds):
    return itertools.chain(*map(lambda d: d.items(), ds))


def merge(*ds):
    return dict(rreduce(fn=lambda l, _: apply(lambda k, v: l + [(k, v)], _),
                        seq=explode(*ds),
                        default=[]))


def assoc(d, k, v):
    return merge(d, {k: v})


def dissoc(d, *ks):
    return keyfilter(lambda x: x not in ks, d)


def merge_with(fn, *ds):
    return rreduce(fn=lambda d, x: apply(lambda k, v:
                                         assoc(d, k, fn(d[k], v)) if k in d else
                                         assoc(d, k, v), x),
                   seq=explode(*ds),
                   default={})


def merge_with_default(fn, default=None, *dicts):
    _fn, _default = fn, default
    return merge_with(_fn, rreduce(fn=lambda d, _: apply(lambda k, v: assoc(d, k, _default), _),
                                   seq=explode(*dicts),
                                   default={}),
                      *dicts)


def assoc_in(d, key_list, val):
    d1 = keys2dict(val, *key_list)
    return recursive_dict_merge(d, d1)


def terminal_dict(*ds):
    return not (are_dicts(*ds) and all(map(lambda x: are_dicts(*x.values()), ds)))


def terminal_dicts(*ds):
    return all(map(terminal_dict, ds))


def update_in(d, key_list, v):
    d1 = keys2dict(v, *key_list)
    return merge_with(lambda a, b: recursive_dict_merge(a, b) if not are_dicts(a, b) else b, d, d1)


def recursive_dict_merge(*ds):
    return merge_with(lambda a, b: recursive_dict_merge(a, b) if not terminal_dicts(a, b) else merge(a, b), *ds)


def keys2dict(val, *ks):
    v_in = reduce(lambda x, y: {y: x}, (list(ks) + [val])[::-1])
    return v_in


def are_instances(items, types):
    return all(map(lambda x: isinstance(x, types), items))


def are_dicts(*ds):
    return are_instances(ds, dict)


def supassoc_in(d, val, k, *ks):
    return merge_with(lambda x, y: merge(x, y) if are_dicts(x.values() + y.values()) else x.values() + y.values(), d,
                      reduce(lambda x, y: {y: x}, ([k] + list(ks) + [val])[::-1]))


def keyfilter(fn, d):
    return reduce(lambda _d, _: apply(lambda k, v: assoc(_d, k, v) if fn(k) else _d, _), d.items(), {})


def valfilter(fn, d):
    return reduce(lambda _d, _: apply(lambda k, v: assoc(_d, k, v) if fn(v) else _d, _), d.items(), {})


def itemfilter(fn, d):
    return reduce(lambda _d, _: apply(lambda k, v: assoc(_d, k, v) if fn(k, v) else _d, _), d.items(), {})


def valmap(fn, d):
    return rreduce(fn=lambda _d, _: apply(lambda k, v: assoc(_d, k, fn(v)), _),
                   seq=d.items(),
                   default={})


def keymap(fn, d):
    return rreduce(fn=lambda _d, _: apply(lambda k, v: assoc(_d, fn(k), v), _),
                   seq=d.items(),
                   default={})


def itemmap(fn, d):
    return rreduce(fn=lambda _d, _: apply(lambda k, v: assoc(_d, *fn(k, v)), _),
                   seq=d.items(),
                   default={})


def nary(fn):
    def _nary(*x):
        if len(x) == 2:
            return fn(*x)
        if len(x) == 1:
            return fn(first(x))
        else:
            return fn(first(x), _nary(*rest(x)))

    return _nary


def append(*seqs):
    return rreduce(fn=lambda x, y: x + y,
                   seq=seqs)


def windows(n, seq):
    return rreduce(
        fn=lambda groups, nxt: (append(groups[:-1],
                                       [append(groups[-1], [nxt])]) if (groups and len(groups[-1]) < n) else
                                append(groups, [[nxt]]) if groups
                                else [[nxt]]),
        seq=seq,
        default=[])
