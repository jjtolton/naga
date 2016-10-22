import itertools
from functools import reduce, partial


from naga.lazy_seq import LazySeq

seq_types = list, tuple, str, LazySeq


def rreduce(fn, seq, default=None):
    """'readable reduce' - More readable version of reduce with arrity-based dispatch; passes keyword arguments
    to functools.reduce"""

    # if two arguments
    if default is None:
        return reduce(fn, seq)

    # if three arguments
    return reduce(fn, seq, default)


def layreduce(fn, seq, default=None):
    """LazySeq version of reduce that returns 1 item """
    def _lzreduce(fn, seq, default=None):
        accum = default if somep(default) else first(seq)
        _seq = LazySeq(seq) if somep(default) else rest(seq)
        while True:
            yield accum
            accum = fn(accum, first(_seq))
            _seq = rest(_seq)
            if not _seq:
                raise StopIteration

    return LazySeq(_lzreduce(fn, seq, default))


def get_in(d, ks, not_found=None):
    """Returns the value in a nested associative structure,
where ks is a sequence of keys. Returns nil if the key
is not present, or the not-found value if supplied.
    :param d: dict
    :param ks: list of keys
    :param not_found: what to return if keys not in d
    :return: val
    """
    return reduce(lambda x, y: x.get(y, {}), ks, d) or not_found


def apply(fn, x):
    """Applies fn to the argument list formed by prepending intervening arguments to args.

    apply(fn, x) --> fn(*x)"""
    return fn(*x)


def dec(n):
    """dec[rement].  Return n - 1"""
    return n - 1


def inc(n):
    """inc[rement].  Return n + 1"""
    return n + 1


def first(iterable):
    """Returns the first item in the collection. If iterable evaluates to None, returns None."""
    return nth(iterable, 0) if iterable else None


def nth(seq, idx):
    """Return the nth item of a sequence.  Constant time if list, tuple, or str;
    linear time if a generator or LazySeq"""
    if isinstance(seq, seq_types):
        return seq[idx]
    return LazySeq(seq)[idx]


def second(seq):
    """Same as first(rest(seq))

    :param seq: sequence or iterable
    :return: val
    """
    return nth(seq, 1)


def third(seq):
    """nth(seq, 2)"""
    return nth(seq, 2)


def fourth(seq):
    """nth(seq, 4)"""
    return nth(seq, 3)


def fifth(seq):
    """nth(seq, 4)"""
    return nth(seq, 4)


def sixth(seq):
    """nth(seq, 5)"""
    return nth(seq, 5)


def seventh(seq):
    """nth(seq, 6)"""
    return nth(seq, 6)


def eigth(seq):
    """nth(seq, 7)"""
    return nth(seq, 8)


def ninth(seq):
    """nth(seq, 8)"""
    return nth(seq, 8)


def tenth(seq):
    """nth(seq, 9)"""
    return nth(seq, 9)


def compose(fns, x=None):
    """Takes a set of functions and returns a fn that is the composition
of those fns.  The returned fn takes a variable number of args,
applies the rightmost of fns to the args, the next
fn (right-to-left) to the result, etc.  If no value is supplied, returns a
stateful transducer"""
    if x is None:
        return partial(compose, fns)

    return rreduce(fn=lambda a, b: b(a),
                   seq=fns,
                   default=x)


def last(iterable):
    """Return the last item in an iterable, in linear time"""
    return LazySeq(iterable)[-1]


def rest(iterable):
    """Returns a LazySeq of the items after the first. Will be an empty list if """
    if len(iterable) == 0:
        return iterable
    return LazySeq(itertools.islice(iterable, 1, None))


def iterate(fn, x):
    """Returns a LazySeq of x, (f x), (f (f x)) etc. f must be free of side-effects"""

    def _iterate(fn, x):
        val = x
        while True:
            yield val
            val = fn(val)

    return LazySeq(_iterate(fn, x))


def take(n, seq=None):
    """Returns a lazy sequence of the first n items in coll, or all items if
there are fewer than n.  Returns a stateful transducer when
no collection is provided."""

    if seq is None:
        return partial(take, n)

    return LazySeq(itertools.islice(iter(seq), 0, n))


def drop(n, seq=None):
    """Returns a lazy sequence of all but the first n items in coll.
Returns a stateful transducer when no collection is provided."""
    if seq is None:
        return partial(n, seq)

    return LazySeq(itertools.islice(seq, n, None))


def explode(*ds):
    """Returns a LazySeq of the concatenated (key,value) pairs of the provided dictionaries"""
    return LazySeq(itertools.chain(*map(lambda d: d.items(), ds)))


def merge(*seqs):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping from
the latter (left-to-right) will be the mapping in the result."""

    return dict(rreduce(fn=lambda l, _: apply(lambda k, v: l + [(k, v)], _),
                        seq=explode(*seqs),
                        default=[]))


def assoc(m, k, v):
    """assoc[iate]. When applied to a map, returns a new map of the
  same (hashed/sorted) type, that contains the mapping of key(s) to
  val(s). When applied to a sequence, returns a new sequence of that
  type that contains val v at index k."""

    if isinstance(m, seq_types):
        return append(m[:k] + [v] + m[k + 1:])

    return merge(m, {k: v})


def dissoc(d, *ks):
    """dissoc[iate]. If d is a dict, returns a new map of the same (hashed/sorted) type,
that does not contain a mapping for key(s).  If d is a str, returns a string without the letters listed as keys.
For any other sequence type, returns a LazySeq with the listed keys filtered."""

    if isinstance(d, str):
        filtered = itertools.ifilter(lambda x: x not in ks, d)
        return ''.join(filtered)

    if isinstance(d, seq_types):
        return LazySeq(itertools.ifilter(lambda x: x not in ks, d))

    return keyfilter(lambda x: x not in ks, d)


def merge_with(fn, *ds):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping(s)
from the latter (left-to-right) will be combined with the mapping in
the result by calling fn(val-in-result, val-in-latter)."""

    return rreduce(fn=lambda d, x: apply(lambda k, v:
                                         assoc(d, k, fn(d[k], v)) if k in d else
                                         assoc(d, k, v), x),
                   seq=explode(*ds),
                   default={})


def merge_with_default(fn, default=None, *dicts):
    """Like merge_with, except all keys are initialized to default value specified to simplify the collision-fn."""
    _fn, _default = fn, default
    return merge_with(_fn, rreduce(fn=lambda d, _: apply(lambda k, v: assoc(d, k, _default), _),
                                   seq=explode(*dicts),
                                   default={}),
                      *dicts)


def assoc_in(d, key_list, val):
    """Associates a value in a nested associative structure, where ks is a
sequence of keys and v is the new value and returns a new nested structure.
If any levels do not exist, hash-maps will be created.  Note that non-destructively merges
keys into dictionaries.  I.e.:

>>> d = {1: {2: {3: 4}}}
>>> assoc_in(d, [1, 2, 'A'], 'X')
{1: {2: {3: 4,
        'A': 'X'}}"""
    d1 = keys2dict(val, *key_list)
    return recursive_dict_merge(d, d1)


def terminal_dict(*ds):
    return not (are_dicts(*ds) and all(map(lambda x: are_dicts(*x.values()), ds)))


def terminal_dicts(*ds):
    return all(map(terminal_dict, ds))


def update_in(d, key_list, fn):
    """'Updates' a value in a nested associative structure, where ks is a
sequence of keys and f is a function that will take the old value
and any supplied args and return the new value, and returns a new
nested structure.  If any levels do not exist, hash-maps will be
created."""

    v = fn(get_in(d, key_list))
    d1 = keys2dict(v, *key_list)
    return merge_with(lambda a, b: recursive_dict_merge(a, b) if not are_dicts(a, b) else b, d, d1)


def recursive_dict_merge(*ds):
    """Recursively merge dictionaries"""
    return merge_with(lambda a, b: recursive_dict_merge(a, b) if not terminal_dicts(a, b) else merge(a, b), *ds)


def keys2dict(val, *ks):
    """Convert a value and a list of keys to a nested dictionary with the value at the leaf"""
    v_in = reduce(lambda x, y: {y: x}, (list(ks) + [val])[::-1])
    return v_in


def are_instances(items, types):
    """Plural is isinstance"""
    return all(map(lambda x: isinstance(x, types), items))


def are_dicts(*ds):
    return are_instances(ds, dict)


def supassoc_in(d, val, k, *ks):
    """Like assoc_in, except collisions are handled by grouping into a list rather than merging"""
    ds = d, keys2dict(val, *itertools.chain([k], ks))
    return recursive_group_dicts(*ds)


def recursive_group_dicts(*ds):
    """Like recursive_dict_merge, except handles recursive collisions by grouping into a list instead of merging"""
    return merge_with(lambda x, y: recursive_group_dicts(x, y) if not terminal_dicts(x, y) else {first(x.keys()): x.values() + y.values()}, *ds)

def keyfilter(fn, d):
    # TODO: make purely functional when persistent vectors are finished
    return {k: v for k, v in d.items() if fn(k)}


def valfilter(fn, d):
    """returns {k: v for k, v in d.items() if fn(v)}"""
    # TODO: make purely functional when persistent vectors are finished
    return {k: v for k, v in d.items() if fn(v)}


def itemfilter(fn, d):
    """returns {k: v for k, v in d.items() if fn(k, v)}"""
    # TODO: make purely functional when persistent vectors are finished
    return {k: v for k, v in d.items() if fn(k, v)}


def valmap(fn, d):
    """{k: fn(v) for k, v in d.items()}"""
    # TODO: make purely functional when persistent vectors are finished
    return {k: fn(v) for k, v in d.items()}


def keymap(fn, d):
    """returns {fn(k): v for k, v in d.items()}"""
    # TODO: make purely functional when persistent vectors are finished
    return {fn(k): v for k, v in d.items()}


def itemmap(fn, d):
    """returns dict(fn(k, v) for k, v in d.items())"""
    # TODO: make purely functional when persistent vectors are finished
    return dict(fn(k, v) for k, v in d.items())


def nary(fn):
    """fn(a, b) --> fn(*x).  Only works if the output type is the same as the
     input type"""

    def _nary(*x):
        if len(x) == 2:
            return fn(*x)
        if len(x) == 1:
            return fn(first(x))
        else:
            return fn(first(x), _nary(*rest(x)))

    return _nary


def append(*seqs):
    """Returns a LazySeq concatenation of iterables (works in constant time)."""
    return LazySeq(itertools.chain(*seqs))


def pop(iterable):
    seq = LazySeq(iterable)
    return LazySeq(itertools.islice(0, len(seq) - 1))


def windows(n, seq):
    return LazySeq(itertools.izip_longest(*(seq[i::n] for i in range(n))))


def conj(seq, *items):
    return itertools.chain(seq, items)


def nonep(x):
    return x is None


def complement(x):
    return not x


def somep(x):
    return complement(nonep(x))
