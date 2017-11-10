import collections
import itertools
from functools import reduce, partial

import types

seq_types = list, tuple, str

nil = -1243901023098409


def rreduce(fn, seq, default=None):
    """'readable reduce' - More readable version of reduce with arrity-based dispatch; passes keyword arguments
    to functools.reduce"""

    # if two arguments
    if default is None:
        return reduce(fn, seq)

    # if three arguments
    return reduce(fn, seq, default)


def reductions(fn, seq, default=nil):
    """LazySeq version of reduce that returns 1 item at a time"""

    if default is nil:
        acc = first(seq)
        x = second(seq)
        s = rest(rest(seq))
    else:
        acc = default
        x = first(seq)
        s = rest(seq)

    yield acc
    while True:
        acc = fn(acc, x)
        x = first(s)
        s = rest(s)
        yield acc
        if len(s) == 0:
            break
    yield fn(acc, x)


def get_in(d, ks, not_found=None):
    """Returns the value in a nested associative structure,
where ks is a sequence of keys. Returns nil if the key
is not present, or the not-found value if supplied.
    :param d: dict
    :param ks: list of keys
    :param not_found: what to return if keys not in d
    :return: val
    """
    res = reduce(lambda x, y: x.get(y, {}), ks, d)
    if res is None:
        return not_found
    else:
        return res



def apply(fn, x):
    """Applies fn to the argument list formed by prepending intervening arguments to args.

    apply(fn, x) --> fn(*x)"""
    return fn(*x)


def some(seq):
    """Returns first truthy value or False"""
    for e in seq:
        if e:
            return e
    return False


def mapv(f, x):
    """Returns a list consisting of the result of applying f to the
set of first items of each coll, followed by applying f to the set
of second items in each coll, until any one of the colls is
exhausted.  Any remaining items in other colls are ignored. Function
f should accept number-of-colls arguments."""
    return list(map(f, x))

def stab(x, *forms):
    """Equivalent to Clojure's -> macro, requires special syntax (see example)
    
Threads the expr through the forms. Inserts x as the
second item in the first form, making a list of it if it is not a
list already. If there are more forms, inserts the first form as the
second item in second form, etc.

Example:
    
>>> stab({}, (assoc, 1, 2), (assoc, 'cat', 'dog'))
{1: 2, 'cat': 'dog'}
"""
    if len(forms) == 0:
        return x
    if not isinstance(first(forms), tuple):
        form = (first(forms),)
    else:
        form = first(forms)

    f = first(form)
    args = (x,) + tuple(rest(form))
    return stab(f(*args), *rest(forms))


# alias in case you don't like the name stab
threadfirst = threadf = stab


def stabb(x, *forms):
    """Equivalent to Clojure's ->> macro, requires special syntax (see example)
    
Threads the expr through the forms. Inserts x as the
last item in the first form, making a list of it if it is not a
list already. If there are more forms, inserts the first form as the
last item in second form, etc.

Example:
>>> stabb(range(10), (filter, lambda x: x % 2 == 0), (map, lambda x: x * 2), list)
[0, 4, 8, 12, 16]
"""
    if len(forms) == 0:
        return x
    if not isinstance(first(forms), tuple):
        form = (first(forms),)
    else:
        form = first(forms)

    f = first(form)
    args = tuple(rest(form)) + (x,)
    return stabb(f(*args), *rest(forms))


# alias in case you don't like the name stabb
threadlast = threadl = stabb


def dec(n):
    """dec[rement].  Return n - 1"""
    return n - 1


def inc(n):
    """inc[rement].  Return n + 1"""
    return n + 1


def first(iterable):
    """Returns the first item in the collection. If iterable evaluates to None, returns None."""
    if isinstance(iterable, seq_types):
        return iterable[0]
    if isinstance(iterable, collections.MutableMapping):
        return dict([first(iterable.items())])
    if isinstance(iterable, types.GeneratorType):
        _, s = itertools.tee(iterable)
        return next(s)


def nth(seq, idx):
    """Return the nth item of a sequence.  Constant time if list, tuple, or str;
    linear time if a generator or LazySeq"""
    _, _seq = itertools.tee(seq)
    if isinstance(seq, seq_types):
        return seq[idx]
    return next(itertools.islice(_seq, idx, idx + 1))


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
fn (left-to-right) to the result, etc.  If no value is supplied, returns a
stateful transducer"""
    if x is None:
        return partial(compose, fns)

    return reduce(lambda a, b: b(a), fns, x)


def comp(*fns):
    """Takes a set of functions and returns a fn that is the composition
of those fns.  The returned fn takes a variable number of args,
applies the rightmost of fns to the args, the next
fn (left-to-right) to the result, etc.  Returns a
stateful transducer"""

    return partial(reduce, lambda a, b: b(a), reversed(fns))


def last(iterable):
    """Return the last item in an iterable, in linear time"""
    if isinstance(iterable, seq_types):
        return iterable[-1]
    if isinstance(iterable, dict):
        return first(iterable)
    if isinstance(iterable, types.GeneratorType):
        _, seq = itertools.tee(iterable)
        return next(itertools.islice(seq, -1, None))


def rest(iterable):
    """Returns a LazySeq of the items after the first. Will be an empty list if iterable is empty generator
    or initial item type if empty iterable"""
    if isinstance(iterable, seq_types):
        return iterable[1:]
    if isinstance(iterable, collections.MutableMapping):
        if len(iterable) == 0:
            return iterable
        return dissoc(iterable, first(iterable.keys()))

    if isinstance(iterable, types.GeneratorType):
        return itertools.islice(iterable, 1)


def iterate(fn, x):
    """Returns a LazySeq of x, (f x), (f (f x)) etc. f must be free of side-effects"""

    def _iterate(fn, x):
        val = x
        while True:
            yield val
            val = fn(val)

    return _iterate(fn, x)


def take(n, seq=None):
    """Returns a lazy sequence of the first n items in coll, or all items if
there are fewer than n.  Returns a stateful transducer when
no collection is provided."""

    if seq is None:
        return partial(take, n)

    return itertools.islice(iter(seq), 0, n)


def drop(n, seq=None):
    """Returns a lazy sequence of all but the first n items in coll.
Returns a stateful transducer when no collection is provided."""
    if seq is None:
        return partial(n, seq)

    return itertools.islice(seq, n, None)


def explode(*ds):
    """Returns a LazySeq of the concatenated (key,value) pairs of the provided dictionaries"""
    return itertools.chain(*map(lambda d: d.items(), ds))


def identity(x):
    return x


def merge(*ds):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping from
the latter (left-to-right) will be the mapping in the result."""
    return dict(itertools.chain(*map(lambda _d: _d.items(), ds)))


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
        filtered = filter(lambda x: x not in ks, d)
        return ''.join(filtered)

    if isinstance(d, seq_types):
        return filter(lambda x: x not in ks, d)

    return keyfilter(lambda x: x not in ks, d)


def merge_with(fn, *ds):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping(s)
from the latter (left-to-right) will be combined with the mapping in
the result by calling fn(val-in-result, val-in-latter)."""
    return reduce(
        lambda d, x: dict(itertools.chain(
            d.items(),
            dict(((k, v) if k not in d else (k, fn(d[k], x[k])) for k, v in x.items())).items())),
        ds)


def merge_with_default(fn, default='null', *dicts):
    """Like merge_with, except all keys are initialized to default value specified to simplify the collision-fn.
    If no default is specified, will use the initial values of the last dictionary merged."""
    return merge_with(fn, valmap(lambda v: v if default is 'null' else default, merge(*dicts)), *dicts)


def assoc_in(d, key_list, val):
    """Associates a value in a nested associative structure, where ks is a
sequence of keys and v is the new value and returns a new nested structure.
If any levels do not exist, hash-maps will be created.  Note that non-destructively merges
keys into dictionaries.  I.e.:

>>> d = {1: {2: {3: 4}}}
>>> assoc_in(d, [1, 2, 'A'], 'X')
{1: {2: {3: 4,
        'A': 'X'}}"""
    if len(key_list) == 1:
        return assoc(d, first(key_list), val)

    if first(key_list) not in d:
        return assoc(d, first(key_list), assoc_in({}, rest(key_list), val))

    return assoc(d, first(key_list), assoc_in(d[first(key_list)], rest(key_list), val))


def terminal_dict(*ds):
    return not (are_dicts(*ds) and all(map(lambda x: are_dicts(*x.values()), ds)))


def terminal_dicts(*ds):
    return all(map(terminal_dict, ds))


def get(d, k):
    return d[k]


def update(d, k, fn, *args, **kwargs):
    return assoc(d, k, fn(get(d, k), *args, **kwargs))


def update_in(d, key_list, fn):
    """'Updates' a value in a nested associative structure, where ks is a
sequence of keys and f is a function that will take the old value
and any supplied args and return the new value, and returns a new
nested structure.  If any levels do not exist, hash-maps will be
created."""
    val = get_in(d, key_list)
    if val is None:
        f = lambda: fn()
    else:
        f = lambda: fn(val)
    return assoc_in(d, key_list, f())


def recursive_dict_merge(*ds):
    """Recursively merge dictionaries"""
    return merge_with(lambda a, b: recursive_dict_merge(a, b) if not terminal_dicts(a, b) else merge(a, b), *ds)


def keys2dict(val, *ks):
    """Convert a value and a list of keys to a nested dictionary with the value at the leaf"""
    v_in = reduce(lambda x, y: {y: x}, (list(ks) + [val])[::-1])
    return v_in


def are_instances(items, types):
    """Plural of isinstance"""
    return all(map(lambda x: isinstance(x, types), items))


def are_dicts(*ds):
    return are_instances(ds, dict)


def supassoc_in(d, val, k, *ks):
    """Like assoc_in, except collisions are handled by grouping into a list rather than merging"""
    ds = d, keys2dict(val, *itertools.chain([k], ks))
    return recursive_group_dicts(*ds)


def recursive_group_dicts(*ds):
    """Like recursive_dict_merge, except handles recursive collisions by grouping into a list instead of merging"""
    return merge_with(lambda x, y: recursive_group_dicts(x, y) if not terminal_dicts(x, y) else {
        first(x.keys()): x.values() + y.values()}, *ds)


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
    return list(itertools.chain(*seqs))


def pop(iterable):
    _, seq = itertools.tee(iterable)
    return itertools.islice(0, len(seq) - 1)


def windows(n, seq):
    """Returns a lazy sequence of lists of n items each"""
    if 'zip_longest' in dir(itertools):
        return itertools.zip_longest(*(seq[i::n] for i in range(n)))
    else:
        return itertools.izip_longest(*(seq[i::n] for i in range(n)))


def partition(n, seq):
    """Returns a lazy sequence of lists of n items each"""
    return windows(n, seq)


def conj(seq, *items):
    return list(itertools.chain(seq, items))


def nonep(x):
    return x is None


def complement(x):
    return not x


def somep(x):
    return complement(nonep(x))


def mapv(fn, *colls):
    return list(map(fn, *colls))

def filterv(fn, *colls):
    return list(filter(fn, *colls))

Result = collections.namedtuple('Result', field_names=['args', 'kwargs'])


def ffirst(x):
    return first(first(x))

