import collections
import heapq
import inspect
import itertools
import re
import types
from functools import partial, reduce

from naga import nil
from naga.utils import Namespaced, decorator

seq_types = list, tuple, str


def gentype(x):
    return hasattr(x, '__iter__') and hasattr(x, '__next__')


def typeq(x):
    return lambda a: isinstance(a, x)


def identity(x):
    return x


@decorator
def message(f):
    def _(x, *args, **kwargs):
        return getattr(x, f.__name__)(x, *args, **kwargs)

    return _


@message
def classify():
    """a generic classification method"""


class PatternMap(list):
    counter = itertools.count()

    def score(self, argtypes):
        def _score(x):
            return x.rank

        return tuple([_score(argtype) for argtype in argtypes])

    def push(self, item):
        argtypes, fn = item
        score = self.score(argtypes)
        heapq.heappush(self, (score, next(PatternMap.counter), item))

    def pop(self):
        return heapq.heappop(self)

    def __iter__(self):
        items = []

        for _ in range(len(self)):
            item = self.pop()
            items.append(item)
            yield item[-1]

        self.extend(items)
        heapq.heapify(self)


@decorator
class Dispatch:
    class ArgType:
        rank = None

        @staticmethod
        def classify(self, x):
            raise NotImplementedError

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    class Any(ArgType):
        rank = float('inf')

        @staticmethod
        def classify(self, x):
            return x is self

        def __call__(self, *args, **kwargs):
            return True

        def __new__(cls, *args, **kwargs):
            return cls

        def __repr__(self):
            return 'Any(*)'

    class Star(Any):
        rank = float('inf')

    class Pred(ArgType):
        @staticmethod
        def classify(self, x):
            raise NotImplementedError

        def __init__(self, f, type=nil):
            self.f = f
            self.t = type

        def __call__(self, arg):
            if self.t is not nil:
                if isinstance(arg, self.t):
                    return self.f(arg)
                else:
                    return False
            else:
                return self.f(arg)

    class Or(set):
        rank = 0

        def __call__(self, x):
            for item in self:
                if item == x:
                    return True
                elif isinstance(self, Dispatch.Iterator) and item(x):
                    return True
            return False

        @staticmethod
        def classify(self, x):
            return isinstance(x, set)

    class OrFn(set):

        rank = 2

        def __call__(self, x):
            for fn in self:
                if fn(x):
                    return True
            return False

        @staticmethod
        def classify(self, x):
            return isinstance(x, list)

    class AndFn(tuple):
        rank = 3

        def __new__(cls, x):
            return super().__new__(cls, tuple(x[0]))

        def __call__(self, x):
            for fn in self:
                if not fn(x):
                    return False
            return True

        @staticmethod
        def classify(self, x):
            return (isinstance(x, list) and
                    len(x) == 1 and
                    isinstance(x[0], list))

    Iterator = types.GeneratorType

    class Regex(Pred):
        rank = 2

        regex_type = type(re.compile(''))

        def __init__(self, s, flags=0, type=str):
            super().__init__(lambda x: re.match(s, x, flags=flags),
                             type=type)

        @staticmethod
        def classify(self, x):
            if isinstance(x, str):
                return True
            elif isinstance(x, bytes):
                return True
            elif isinstance(x, self.regex_type):
                return True
            return False

    class Type(Pred):
        rank = 5

        @staticmethod
        def classify(self, x):
            return True

        def __init__(self, x):
            super().__init__(constantly(True), x)

        def __repr__(self):
            return 'Type({})'.format(self.t)

    def __init__(self, f=identity):
        self.f = f
        self.pattern_map = PatternMap()
        self.maxlen = 0
        self.default = f

    def __mul__(self, other):
        return partial(apply, other)

    def classify(self,
                 argtypes,
                 classes=(AndFn, Or, OrFn, Regex, Any, Star, Type)):

        clss = []

        for argtype in argtypes:
            for cls in classes:
                if classify(cls, argtype):
                    clss.append(cls(argtype))
                    break
            else:
                raise TypeError(
                    "Unsupported type invocation: {}".format(argtype))

        return tuple(clss)

    def pattern(self, *argtypes):
        @decorator
        def _dispatch(f):
            self.pattern_map.push((self.classify(argtypes), f))
            self.maxlen = argmax(self.pattern_map, key=lambda x: len(x[0]))
            return self

        return _dispatch

    def declare(self, f):
        argspec = inspect.getfullargspec(f)
        args = argspec.args
        anns = argspec.annotations
        varargs = [Dispatch.Star] if argspec.varargs else []
        return_fn = anns.get('return')
        if return_fn:
            fn = lambda *args, **kwargs: return_fn(f(*args, **kwargs))
        else:
            fn = f

        argvals = [anns.get(arg, Dispatch.Any) for arg in args]
        fout = self.pattern(
            *[*argvals, *varargs])(fn)
        return fout

    def __call__(self, *args, **kwargs):

        def find(args, n=self.maxlen):
            if n < 0:
                return self.default

            for argtypes, fn in list(self.pattern_map):

                if len(argtypes) < len(args[:n]):
                    continue
                elif len(argtypes) == 0 and len(args) == 0:
                    return fn
                elif ((len(argtypes) > len(args)) and
                              Dispatch.Star not in argtypes):
                    continue

                for argtype, arg in itertools.zip_longest(argtypes, args[:n]):

                    if not argtype(arg):
                        break
                else:
                    return fn
            else:
                return find(args, dec(n))

        return find(args)(*args, **kwargs)

    def __repr__(self):
        return 'Dispatch({})'.format(self.f.__name__)


def reductions(fn, seq, default=nil):
    """generator version of reduce that returns 1 item at a time"""

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


def cond(x, *forms):
    if len(forms) % 2 != 0:
        raise Exception("cond requires an even number of forms!")
    for t, a in partition(2, forms):
        if t(x):
            return a


def get_in(d, ks, not_found=None):
    """Returns the value in a nested associative structure,
where ks is a sequence of keys. Returns nil if the key
is not present, or the not-found value if supplied.
    :param d: dict
    :param ks: list of keys
    :param not_found: what to return if keys not in d
    :return: val
    """
    if len(ks) == 1:
        return get(d, first(ks), not_found)
    else:
        return get_in(get(d, first(ks)), rest(ks), not_found)


def apply(fn, *x):
    """Applies fn to the argument list formed by prepending intervening arguments to args.

    apply(fn, x) --> fn(*x)"""
    if len(x) > 0 and isinstance(x[-1], (tuple, list)):
        return apply(fn, *(x[:-1] + tuple(x[-1])))
    else:
        return fn(*x)


def some(fn, seq=nil):
    """Returns first truthy value or False.  Can accept a predicate as first argument."""

    if seq is nil:
        seq = fn
        fn = identity
        return some(fn, seq)

    for e in seq:
        if fn(e):
            return e
    return False


def mapv(f, *colls):
    """Returns a list consisting of the result of applying f to the
set of first items of each coll, followed by applying f to the set
of second items in each coll, until any one of the colls is
exhausted.  Any remaining items in other colls are ignored. Function
f should accept number-of-colls arguments."""
    return list(map(f, *colls))


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


@Dispatch
def first():
    """Returns the first item in the collection. If iterable evaluates to None, returns None."""


def nth(seq, idx):
    """Return the nth item of a sequence.  Constant time if list, tuple, or str;
    linear time if a generator"""
    return get(seq, idx)


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


def compose(fns, x=nil):
    """Takes a set of functions and returns a fn that is the composition
of those fns.  The returned fn takes a variable number of args,
applies the rightmost of fns to the args, the next
fn (left-to-right) to the result, etc.  If no value is supplied, returns a
stateful transducer"""
    if x is nil:
        return partial(compose, fns)

    return reduce(lambda a, b: b(a), fns, x)


def comp(*fns):
    """Takes a set of functions and returns a fn that is the composition
of those fns.  The returned fn takes a variable number of args,
applies the rightmost of fns to the args, the next
fn (left-to-right) to the result, etc.  Returns a
stateful transducer

>>> comp(list, partial(map, inc), partial(map, lambda x: x * 2))([0, 1, 2])
[1, 3, 5]
"""

    return compose(reversed(fns))


def juxt(*fns):
    """ Takes a set of functions and returns a fn that is the juxtaposition
  of those fns.  The returned fn takes a variable number of args, and
  returns a vector containing the result of applying each fn to the
  args (left-to-right)

  >>> juxt(first, second, last)(list(range(10)))
  [0, 1, 9]"""

    return lambda x: [f(x) for f in fns]


@Dispatch
def last(): """Return the last item in an iterable, in linear time"""


@Dispatch
def rest():
    """Returns a the rest of the items after the first.
    Will be an empty list if iterable is empty generator
    or initial item type if empty iterable"""


def iterate(fn, x):
    """Returns a generator of x, (f x), (f (f x)) etc"""

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
        return partial(drop, n)

    return itertools.islice(seq, n, None)


def gtake(n, seq=None):
    "A greedy form of take"
    return list(take(n, seq))


def gdrop(n, seq=None):
    "A greedy form of drop"
    return list(drop(n, seq))


def explode(*ds):
    """Returns a generator of the concatenated (key,value) pairs of the provided dictionaries"""
    return itertools.chain(*map(lambda d: d.items(), ds))


def merge(*ds):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping from
the latter (left-to-right) will be the mapping in the result."""
    return dict(itertools.chain(*(d.items() for d in ds)))


@Dispatch
def assoc():
    """assoc[iate]. When applied to a map, returns a new map of the
  same (hashed/sorted) type, that contains the mapping of key(s) to
  val(s). When applied to a sequence, returns a new sequence of that
  type that contains val v at index k."""


@Dispatch
def dissoc():
    """dissoc[iate]. If d is a dict, returns a new map of the same (hashed/sorted) type,
that does not contain a mapping for key(s).  If d is a str, returns a string without the letters listed as keys.
For any other sequence type, returns a generator with the listed keys filtered."""


def merge_with(fn, *ds):
    """Returns a dict that consists of the rest of the dicts merged onto
the first.  If a key occurs in more than one dict, the mapping(s)
from the latter (left-to-right) will be combined with the mapping in
the result by calling fn(val-in-result, val-in-latter)."""
    return reduce(
        lambda d, x: merge(d, dict(
            ((k, v) if k not in d else (k, fn(d[k], x[k])) for k, v in
             x.items()))),
        ds)


def merge_with_default(fn, default=nil, *dicts):
    """Like merge_with, except all keys are initialized to default value specified to simplify the collision-fn.
    If no default is specified, will use the initial values of the last dictionary merged."""
    return merge_with(fn, valmap(lambda v: v if default is nil else default,
                                 merge(*dicts)), *dicts)


def assoc_in(d, key_list, val):
    """Associates a value in a nested associative structure, where ks is a
sequence of keys and v is the new value and returns a new nested structure.
If any levels do not exist, hash-maps will be created.  Note that non-destructively merges
keys into dictionaries.  I.e.:

>>> d = {1: {2: {3: 4}}}
>>> assoc_in(d, [1, 2, 'A'], 'X')
{1: {2: {3: 4, 'A': 'X'}}}"""
    if len(key_list) == 1:
        return assoc(d, first(key_list), val)

    if first(key_list) not in d:
        return assoc(d, first(key_list),
                     assoc_in(d.__class__({}), rest(key_list), val))

    return update(d, first(key_list), assoc_in, rest(key_list), val)


def terminal_dict(*ds):
    return not (
        are_dicts(*ds) and all(map(lambda x: are_dicts(*x.values()), ds)))


def terminal_dicts(*ds):
    return all(map(terminal_dict, ds))


@Dispatch
def get():
    """Get key "k" from collection "x".

    >>> get({1: 2}, 1)
    2
    >>> get([1, 2], 0)
    1
    >>> get((x for x in range(10)), 2)
    2
    >>> get((1, 2), 0)
    1
    >>> get('abc', 2)
    'c'

    :param x:
    :param k:
    :param not_found:
    :return:
    """


@Dispatch
def update(d, k, fn, *args, **kwargs):
    return d.update(d, k, fn, *args, **kwargs)


def fpartial(f, *args, **kwargs):
    return lambda x: f(*((x,) + args), **kwargs)


# noinspection PyMethodParameters
class Protocol(Namespaced):
    def update(d, k, fn, *args, **kwargs):
        raise NotImplementedError

    def get(d, k, not_found=None):
        raise NotImplementedError

    def first(d):
        try:
            return next(iter(d))
        except StopIteration:
            return []

    def rest(d):
        raise NotImplementedError

    def last(d):
        raise NotImplementedError

    def dissoc(d, *ks):
        raise NotImplementedError


# noinspection PyMethodParameters
class Dict(Protocol):
    def first(d):
        return next(iter(k for k in d))

    def update(d, k, fn, *args, **kwargs):
        return {a: b for a, b in itertools.chain(d.items(), [
            (k, fn(get(d, k), *args, **kwargs))])}

    def get(d, k, not_found=None):
        return d.get(k, not_found)

    def last(d):
        return Dict.first(d)

    def rest(d):
        return list(d)[1:] or {}

    def dissoc(d, *ks):
        ks = set(ks)
        return keyfilter(lambda x: x not in ks, d)

    def assoc(self, k, v):
        return {**self, **{k: v}}


# noinspection PyMethodParameters
class List(Protocol):
    def get(d, k, not_found=None):
        if not 0 <= k <= len(d):
            if not_found is nil:
                return None
            return not_found

        return d[k]

    def update(d, k, fn, *args, **kwargs):
        return d[:k] + [fn(get(d, k), *args, **kwargs)] + d[k + 1:]

    def last(d):
        return d[-1]

    def rest(d):
        return d[1:]

    def dissoc(d, *ks):
        ks = set(ks)
        return filterv(lambda x: x not in ks, d)

    def assoc(self, k, v):
        return [*self[:k], v, *self[k + 1]]


# noinspection PyMethodParameters
class Tuple(Protocol):
    def dissoc(d, *ks):
        ks = set(ks)
        return tuple(filter(lambda x: x not in ks, d))

    def get(d, k, not_found=None):
        return d[k]

    def update(d, k, fn, *args, **kwargs):
        return d[:k] + (fn(get(d, k), *args, **kwargs),) + d[k + 1:]

    def last(d):
        return d[-1]

    def rest(d):
        return d[1:]


class String(Protocol):
    def last(d):
        return d[-1]

    def get(d, k, not_found=None):
        return d[k]

    def update(d, k, fn, *args, **kwargs):
        return d[:k] + fn(get(d, k), *args, **kwargs) + d[k + 1]

    def rest(d):
        return d[1:]

    def dissoc(d, *ks):
        return ''.join(filter(lambda x: x not in ks, d))


# noinspection PyMethodParameters
class Iterable(Protocol):
    def last(d):
        try:
            for x in d:
                pass
            return x
        except NameError:
            return None

    def first(d):
        try:
            return next(iter(d))
        except StopIteration:
            return None

    def rest(d):
        try:
            next(d)
            return d
        except StopIteration:
            return []

    def get(d, k, not_found=None):
        if k < 0 or not isinstance(k, int):
            raise Exception(
                "\"k\" must be an index value greater than one, not {k}(type(k))")
        x = iter(d)
        while True:
            if k == 0:
                return next(x)
            k -= 1
            next(x)


def constantly(x):
    return lambda *args, **kwargs: x


# noinspection PyMethodParameters
class Set(Protocol):
    def dissoc(s, x):
        return s - {x}

    def last(s):
        if len(s) > 0:
            for x in s:
                pass
            return x
        return None

    def first(s):
        if len(s) > 0:
            for x in s:
                return x
        return None

    def update(s, k, fn, *args, **kwargs):
        return (s - {k}) | {fn(k, *args, **kwargs)}


def update_in(d, key_list, fn, *args, **kwargs):
    """'Updates' a value in a nested associative structure, where ks is a
sequence of keys and f is a function that will take the old value
and any supplied args and return the new value, and returns a new
nested structure.  If any levels do not exist, hash-maps will be
created."""
    if len(key_list) == 1:
        return update(d, first(key_list), fn, *args, **kwargs)

    return update(d, first(key_list), update_in, rest(key_list), fn, *args,
                  **kwargs)


class _Reflect:
    vals = {dict: Dict, list: List, str: String, tuple: Tuple, set: Set,
            types.GeneratorType: Iterable}

    @classmethod
    def reflect(cls, x):
        datatype = cls.vals.get(type(x))
        if datatype is None:
            datatype = cond(x,
                            fpartial(isinstance, collections.MutableMapping),
                            Dict,
                            fpartial(isinstance, types.GeneratorType),
                            Iterable,
                            constantly(True), x)
        return datatype


_reflect = _Reflect.reflect


def recursive_dict_merge(*ds):
    """Recursively merge dictionaries"""
    return merge_with(
        lambda a, b: recursive_dict_merge(a, b) if not terminal_dicts(a,
                                                                      b) else merge(
            a, b), *ds)


deep_merge = recursive_dict_merge


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
    return merge_with(
        lambda x, y: recursive_group_dicts(x, y) if not terminal_dicts(x,
                                                                       y) else {
            first(x.keys()): x.values() + y.values()}, *ds)


def keyfilter(fn, d):
    return {k: v for k, v in d.items() if fn(k)}


def valfilter(fn, d):
    """returns {k: v for k, v in d.items() if fn(v)}"""
    return {k: v for k, v in d.items() if fn(v)}


def itemfilter(fn, d):
    """returns {k: v for k, v in d.items() if fn(k, v)}"""
    return {k: v for k, v in d.items() if fn(k, v)}


def valmap(fn, d):
    """{k: fn(v) for k, v in d.items()}"""
    return {k: fn(v) for k, v in d.items()}


def keymap(fn, d):
    """returns {fn(k): v for k, v in d.items()}"""
    return {fn(k): v for k, v in d.items()}


def itemmap(fn, d):
    """returns dict(fn(k, v) for k, v in d.items())"""
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
    """Join sequences"""
    return list(itertools.chain(*seqs))


def pop(iterable):
    it = iter(iterable)
    val = first(iterable)
    while val:
        try:
            yield val
            val = next(it)
        except StopIteration:
            pass


def popv(iterable):
    return list(pop(iterable))


def partition(n, seq):
    """Returns a lazy sequence of lists of n items each"""
    if 'zip_longest' in dir(itertools):
        return itertools.zip_longest(*(seq[i::n] for i in range(n)))
    else:
        return itertools.izip_longest(*(seq[i::n] for i in range(n)))


def conj(x, *args):
    return append(x, args)


class fconj:
    """Fast conj.  Only realizes to list when explicitly iterated over.  """
    __slots__ = ['seq', 'items']

    def __init__(self, seq, *items):
        self.seq = seq
        self.items = items

    def __iter__(self):
        return itertools.chain(self.seq, self.items)


def nonep(x):
    return x is None


@decorator
def complement(f):
    def _complement(*args, **kwargs):
        return not f(*args, **kwargs)

    return _complement


def somep(x):
    return complement(nonep(x))


def filterv(fn, *colls):
    """A greedy version of filter."""
    return list(filter(fn, *colls))


def ffirst(x):
    return first(first(x))


def case(x, *forms):
    """
    >>> case(5, 1, 'one', 3, 'three', 5, 'five')
    'five'
    """
    for a, b in partition(2, forms):
        if a == x:
            return b


def remove(f, x):
    """Remove elements from iterable based on predicate f.
    >>> list(remove(lambda x: x > 2, range(10)))
    [0, 1, 2]"""
    return filter(complement(f), x)


def interleave(*xs):
    """
    >>> x = interleave(range(3), range(3), range(3))
    >>> list(x)
    [0, 0, 0, 1, 1, 1, 2, 2, 2]
    """

    return (a for b in zip(*xs) for a in b)


def intereleavev(*xs):
    return list(interleave(*xs))


def repeatedly(x, times=None):
    """

    >>> r = repeatedly(6, times=7)
    >>> sum(r)
    42
    """
    return itertools.repeat(x, times=times)


def argmax(x, key=identity):
    """
    >>> argmax(['cat', 'rats', 'at'], key=len)
    4

    >>> argmax(['cat', 'rats', 'at'], key=lambda x: x.count('a'))
    1
    """
    return key(max(x, key=key))


def argmin(x, key=identity):
    """Opposite of argmax"""
    return key(min(x, key=key))


def argsort(x, key=identity, reverse=False):
    """Like sorted, but converts using key

    >>> argsort(['cat', 'hats', 'rats'], key=len)
    [3, 4, 4]

    >>> argsort(['cat', 'hats', 'rats'], key=len, reverse=True)
    [4, 4, 3]
    """
    return sorted((key(xi) for xi in x), reverse=reverse)


#################
# instantiation #
#################

def _instantiate(f, datatypes=(
        (Dict, dict), (List, list), (Set, set), (Tuple, tuple),
        (String, str))):
    for protocol, datatype in datatypes:
        if hasattr(protocol, f.__name__):
            f.pattern(datatype)(getattr(protocol, f.__name__))


critical_fns = mapv(_instantiate,
                    [first, last, rest, assoc, dissoc, get, update])
