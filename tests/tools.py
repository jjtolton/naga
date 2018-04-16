import itertools
import operator
import unittest
from functools import reduce, lru_cache, partial

from naga.tools import apply, merge, assoc, dissoc, merge_with, \
    merge_with_default, assoc_in, update_in, terminal_dict, \
    windows, append, explode, conj, first, nth, rest, reductions, take, \
    iterate, drop, get, Dispatch, identity


class FunkyToolsTest(unittest.TestCase):
    def setUp(self):
        self.ds = a, b, c = [
            {1: 2,
             3: 4,
             4: 5},
            {'a': 'b',
             'c': 'd',
             'e': 'f'},
            {'name': 'taco'}
        ]

    def test_explode_convs_list_dicts_to_list_tuples(self):
        ds = self.ds
        expected_res = {(1, 2), (3, 4), (4, 5), ('a', 'b'), ('c', 'd'),
                        ('e', 'f'), ('name', 'taco')}
        res = set(apply(explode, ds))
        self.assertEqual(expected_res, res)

    def test_merge_merges_dicts(self):
        ds = self.ds
        expected_res = {
            1: 2,
            3: 4,
            4: 5,
            'a': 'b',
            'c': 'd',
            'e': 'f',
            'name': 'taco'
        }

        res = merge(*ds)

        self.assertEqual(expected_res, res)

    def test_assoc_assocs_kvs_into_dict(self):
        _, _, c = self.ds

        expected_res = {'name': 'taco', 'num': 42}
        res = assoc(c, 'num', 42)
        self.assertEqual(expected_res, res)

    def test_dissoc_removes_kvs_from_dict(self):
        a, _, _ = self.ds
        expected_res = {3: 4, 4: 5}
        res = dissoc(a, 1)
        self.assertEqual(expected_res, res)

    def test_merge_with_merges_only_ints(self):
        a, _, c = self.ds
        expected_res = {1: 4, 3: 8, 4: 10, 'name': 'taco'}
        res = merge_with(lambda x, y: x + y if (
            isinstance(x, int) and isinstance(y, int)) else y, a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_doubles_strings_only(self):
        a, _, c = self.ds
        expected_res = {1: 2, 3: 4, 4: 5, 'name': 'tacotaco'}
        res = merge_with(lambda x, y: x + y if isinstance(x, str) else y,
                         a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_default_cats_to_list(self):
        ds = self.ds
        expected_res = {1: [2, 2], 3: [4, 4], 4: [5, 5], 'a': ['b', 'b'],
                        'c': ['d', 'd'], 'e': ['f', 'f'],
                        'name': ['taco', 'taco']}
        res = merge_with_default(lambda x, y: x + [y], [], *(ds + ds))
        self.assertEqual(expected_res, res)

    def test_merge_with_default_sums_or_adds(self):
        a = first(self.ds)
        expected_res = {1: 4, 3: 8, 4: 10}
        res = merge_with_default(lambda x, y: x + y, 0, *[a, a])
        self.assertEqual(expected_res, res)

    def test_assoc_in_non_destructively_adds_key_to_path(self):
        _, _, c = self.ds
        expected_res = {'name': 'taco',
                        'feelings': {'they': {'are': 'mutual'}}}

        res = assoc_in(c, ['feelings', 'they', 'are'], 'mutual')
        self.assertEqual(expected_res, res)

    def test_assoc_in_merges_when_same_level_as_nonterminal_dict(self):
        _, _, c = self.ds
        initial_d = assoc_in(c, ['thisis'], {'a': dict})
        expected_res = {'name': 'taco',
                        'thisis': {'a': dict,
                                   'b': dict}
                        }
        res = assoc_in(initial_d, ['thisis', 'b'], dict)
        self.assertEqual(expected_res, res)

    def test_assoc_in_works_with_mixed_datatypes(self):
        d0 = {'a': {'c': [1, 2, 3],
                    'd': {'e': [4, 5, 6]}}}

        expected_res = {'a': {'c': [1, 2, 3],
                              'd': {'e': [4, 5, 6],
                                    'f': {'g': [9, 10, 11]}}}}

        res = assoc_in(d0, 'adfg', [9, 10, 11])
        self.assertEqual(expected_res, res)

    def test_assoc_in_adds_dict_okay(self):
        d0 = {'a': {'c': [1, 2, 3],
                    'd': {'e': [4, 5, 6]}}}
        expected_res = {'a': {'c': [1, 2, 3],
                              'd': {'e': [4, 5, 6],
                                    'f': {'1': '2', '3': 'hello'}}}}

        res = assoc_in(d0, ['a', 'd', 'f'], {'1': '2', '3': 'hello'})
        self.assertEqual(expected_res, res)

    def test_update_in_destructively_adds_key_to_path(self):
        _, _, c = self.ds
        initial_d = assoc_in(c, ['thisis'], {'a': dict})
        expected_res = {'name': 'taco',
                        'thisis': {'b': dict}}
        res = update_in(initial_d, ['thisis'], lambda x: {'b': dict})
        self.assertEqual(expected_res, res)

    def test_terminal_dicts_id_terminal_dicts_correctly(self):
        a = {'a': ['this', 'is', 'terminal']}
        b = {'b': {'this': 'ain\'t'}}
        self.assertTrue(terminal_dict(a))
        self.assertFalse(terminal_dict(b))

    def test_windows_creates_groups_of_3s(self):
        lst = range(1, 13)
        expected_res = [(1, 2, 3),
                        (4, 5, 6),
                        (7, 8, 9),
                        (10, 11, 12)]
        res = list(windows(3, lst))
        self.assertEqual(expected_res, res)

    def test_windows_creates_groups_of_1s(self):
        lst = range(5)
        expected_res = [(x,) for x in lst]
        res = list(windows(1, lst))
        self.assertEqual(expected_res, res)

    def test_append_concatenates_lists(self):
        lsts = range(10), range(10), range(10)
        expected_res = reduce(lambda x, y: list(x) + list(y), lsts)
        res = list(append(*map(list, lsts)))
        self.assertEqual(expected_res, res)

    def test_conjugate_conjugates_into_lst(self):
        lst = list('abc')
        appendos = 'def'
        expected_res = list('abcdef')
        res = list(conj(lst, *appendos))
        self.assertEqual(expected_res, res)

    def test_first_gets_first_item_of_list(self):
        a = [1]
        expected_res = 1
        res = first(a)
        self.assertEqual(expected_res, res)

    def test_first_gets_first_item_of_tuple(self):
        a = 1,
        expected_res = 1
        res = first(a)
        self.assertEqual(expected_res, res)

    def test_nth_returns_nth_item(self):
        a = [1, 2, 3]
        expected_res = 3
        res = nth(a, 2)
        self.assertEqual(expected_res, res)

    def test_rest_returns_none_on_empty_seq(self):
        seqs = [], tuple(), {}, ''
        expected_res = list(seqs)
        res = [rest(x) for x in seqs]
        self.assertEqual(expected_res, res)

    def test_layreduce_sums_numbers(self):
        nums = list(range(5))
        expected_res = [0, 1, 3, 6, 10]
        res = list(reductions(fn=operator.add,
                              seq=nums))
        self.assertEqual(expected_res, res)

    def test_first_returns_first_on_str(self):
        a = 'hey'
        expected_res = 'h'
        res = first(a)
        self.assertEqual(expected_res, res)

    def test_iterate_iterates_over_fn(self):
        double = lambda x: x * 2
        expected_res = [2, 4, 8, 16, 32]
        res = list(take(5, iterate(double, 2)))
        self.assertEqual(expected_res, res)

    def test_take_takes_first_10_of_inf_seq(self):
        expected_res = list(range(10))
        res = list(take(10, itertools.count()))
        self.assertEqual(expected_res, res)

    def test_drop_drops_first_5_of_inf_seq(self):
        expected_res = list(range(5, 10))
        res = list(take(5, drop(5, itertools.count())))
        self.assertEqual(expected_res, res)


class FuncyToolsTest2(unittest.TestCase):
    def setUp(self):
        self.ds = a, b, c = [
            {1: 2,
             3: 4,
             4: 5},
            {'a': 'b',
             'c': 'd',
             'e': 'f'},
            {'name': 'taco'}
        ]

    def test_explode_convs_list_dicts_to_list_tuples(self):
        ds = self.ds
        expected_res = {(1, 2), (3, 4), (4, 5), ('a', 'b'), ('c', 'd'),
                        ('e', 'f'), ('name', 'taco')}
        res = set(explode(*ds))
        self.assertEqual(expected_res, res)

    def test_merge_merges_dicts(self):
        ds = self.ds
        expected_res = {
            1: 2,
            3: 4,
            4: 5,
            'a': 'b',
            'c': 'd',
            'e': 'f',
            'name': 'taco'
        }

        res = merge(*ds)

        self.assertEqual(expected_res, res)

    def test_assoc_assocs_kvs_into_dict(self):
        _, _, c = self.ds

        expected_res = {'name': 'taco', 'num': 42}
        res = assoc(c, 'num', 42)
        self.assertEqual(expected_res, res)

    def test_dissoc_removes_kvs_from_dict(self):
        a, _, _ = self.ds
        expected_res = {3: 4, 4: 5}
        res = dissoc(a, 1)
        self.assertEqual(expected_res, res)

    def test_merge_with_merges_only_ints(self):
        a, _, c = self.ds
        expected_res = {1: 4, 3: 8, 4: 10, 'name': 'taco'}
        res = merge_with(lambda x, y: x + y if (
            isinstance(x, int) and isinstance(y, int)) else y, a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_doubles_strings_only(self):
        a, _, c = self.ds
        expected_res = {1: 2, 3: 4, 4: 5, 'name': 'tacotaco'}
        res = merge_with(lambda x, y: x + y if isinstance(x, str) else y,
                         a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_default_cats_to_list(self):
        ds = self.ds
        expected_res = {1: [2, 2], 3: [4, 4], 4: [5, 5], 'a': ['b', 'b'],
                        'c': ['d', 'd'], 'e': ['f', 'f'],
                        'name': ['taco', 'taco']}
        res = merge_with_default(lambda x, y: x + [y], [], *(ds + ds))
        self.assertEqual(expected_res, res)

    def test_merge_with_default_sums_or_adds(self):
        a = self.ds[0]

        expected_res = {1: 4, 3: 8, 4: 10}
        res = merge_with_default(lambda x, y: x + y, 0, *[a, a])
        self.assertEqual(expected_res, res)

    def test_assoc_in_non_destructively_adds_key_to_path(self):
        _, _, c = self.ds
        expected_res = {'name': 'taco',
                        'feelings': {'they': {'are': 'mutual'}}}

        res = assoc_in(c, ['feelings', 'they', 'are'], 'mutual')
        self.assertEqual(expected_res, res)

    def test_assoc_in_merges_when_same_level_as_nonterminal_dict(self):
        _, _, c = self.ds
        initial_d = assoc_in(c, ['thisis'], {'a': dict})
        expected_res = {'name': 'taco',
                        'thisis': {'a': dict,
                                   'b': dict}
                        }
        res = assoc_in(initial_d, ['thisis', 'b'], dict)
        self.assertEqual(expected_res, res)

    def test_assoc_in_works_with_mixed_datatypes(self):
        d0 = {'a': {'c': [1, 2, 3],
                    'd': {'e': [4, 5, 6]}}}

        expected_res = {'a': {'c': [1, 2, 3],
                              'd': {'e': [4, 5, 6],
                                    'f': {'g': [9, 10, 11]}}}}

        res = assoc_in(d0, 'adfg', [9, 10, 11])
        self.assertEqual(expected_res, res)

    def test_assoc_in_adds_dict_okay(self):
        d0 = {'a': {'c': [1, 2, 3],
                    'd': {'e': [4, 5, 6]}}}
        expected_res = {'a': {'c': [1, 2, 3],
                              'd': {'e': [4, 5, 6],
                                    'f': {'1': 2, '3': 'hello'}}}}

        res = assoc_in(d0, ['a', 'd', 'f'], {'1': 2, '3': 'hello'})
        self.assertEqual(expected_res, res)

    def test_update_in_destructively_adds_key_to_path(self):
        _, _, c = self.ds
        initial_d = assoc_in(c, ['thisis'], {'a': dict})
        expected_res = {'name': 'taco',
                        'thisis': {'b': dict}}
        res = update_in(initial_d, ['thisis'], lambda x: {'b': dict})
        self.assertEqual(expected_res, res)

    def test_terminal_dicts_id_terminal_dicts_correctly(self):
        a = {'a': ['this', 'is', 'terminal']}
        b = {'b': {'this': 'ain\'t'}}
        self.assertTrue(terminal_dict(a))
        self.assertFalse(terminal_dict(b))

    def test_dispatch_acts_as_expected(self):
        @Dispatch
        def foo(*_, **__):
            raise TypeError("Unsupported Types")

        @foo.pattern(foo.Just(0), foo.star)
        def foo(a, *args):
            return a

        @foo.pattern(int, int, int, foo.star)
        def foo(a, b, c, *args):
            return a + b + c * sum(args)

        @foo.pattern(int, int, int)
        def foo(*args):
            return sum(args)

        @foo.pattern(int, int)
        def foo(a, b):
            return a * b

        @foo.pattern(int)
        def foo(a):
            return a + a

        @foo.pattern(foo.regex('.*cat'))
        def foo(cat):
            return f'there\'s a "cat" in {cat}'

        class FooMsg:
            def __init__(self, msg):
                self.msg = msg

            def __repr__(self):
                return f'FooMsg(self.msg)'

        @foo.pattern(FooMsg)
        def foo(msg):
            return repr(msg)

        @foo.pattern(foo.GeneratorType)
        def foo(args):
            return foo(*args)

        @foo.pattern(type(range(0)))
        def foo(args):
            return foo(*args)

        @Dispatch
        def fib(n):
            return fib(n - 2) + fib(n - 1)

        @fib.pattern(fib.pred(lambda x: x == 1 or x == 0))
        def fib(n):
            return n

        self.assertEqual(fib(10), 55)

        @foo.pattern(list, int)
        def foo(args, n):
            return get(args, n)

        self.assertEqual(foo(list(range(10)), 2), 2)

        self.assertEqual(foo(1, 2, 3, 4), 15)
        self.assertEqual(foo(1, 2, 3), 6)
        self.assertEqual(foo(1, 1), 1)
        self.assertEqual(foo(1), 2)
        self.assertEqual(foo('cat'), """there's a "cat" in cat""")
        foomsg = FooMsg('name=taco')
        self.assertEqual(foo(foomsg),
                         repr(foomsg))
        self.assertEqual(foo(range(3)), 3)
        self.assertEqual(foo(x for x in range(3)), 3)
        self.assertEqual(foo(0), 0)
        self.assertEqual(foo(0, 1), 0)

        with self.assertRaises(TypeError):
            foo('dogs are great!')

    def test_dispatch_on_recursion(self):
        @Dispatch
        def fib():
            pass

        @fib.pattern(int)
        @lru_cache(None)
        def fib(n):
            return fib(n - 2) + fib(n - 1)

        @fib.pattern(dict)
        def fib(d):
            return fib(d['n'] - 2) + fib(d['n'] - 1)

        @fib.pattern(fib.Or(fib.Just(1), fib.Just(0)))
        @lru_cache(None)
        def fib(n):
            return n

        self.assertEqual(fib({'n': 100}), 354224848179261915075)

    def test_dispatch_on_arrity(self):

        @Dispatch
        def foo():
            pass

        @foo.pattern(int)
        def foo(x):
            return foo(1, x)

        @foo.pattern(int, int)
        def foo(a, b):
            return foo(a + b, a, b)

        @foo.pattern(int, int, int)
        def foo(*args):
            return reduce(operator.mul, args)

        @foo.pattern(type(range(0)))
        def foo(args):
            return foo(*args)

        self.assertEqual(foo(2), 6)
        self.assertEqual(foo(2), 6)
        self.assertEqual(foo(1, 2, 3, 4), 24)
        self.assertEqual(foo(range(2, 3)), 6)
        self.assertEqual(foo(range(1, 2)), 2)
        self.assertEqual(foo(range(1000)), 0)
        self.assertEqual(foo(*range(1000)), 0)

    def test_dispatch_with_generator_types(self):

        @Dispatch
        def foo():
            for n in itertools.count():
                yield n

        @foo.pattern(list, int, foo.Or(foo.GeneratorType, foo.Just(None)))
        def foo(l, n, ns=None):
            if ns is None:
                ns = foo()

            if len(l) > n:
                return l

            return foo(l + [next(ns)], n, ns)

        @foo.pattern(int)
        def foo(n):
            return foo([], n, foo())

    def test_declare_for_pattern_registration(self):

        @Dispatch
        def foo(*args):
            raise TypeError("Unhandled type(s)")

        @foo.declare
        def int_generator():
            for n in itertools.count():
                yield n

        @foo.declare
        def accumulate_list(
                x: list,
                n: int,
                ns: foo.Or(foo.GeneratorType,
                           foo.Just(None)) = None):
            if ns is None:
                ns = foo()

            if len(x) > n:
                return x

            return foo(x + [next(ns)], n, ns)

        @foo.declare
        def initiate_list(n: int):
            return foo([], n, foo())

        @foo.declare
        def make_sum_string(x: str, *args):
            return x.format(sum(args))

        self.assertEqual(foo(10), list(range(11)))

        @foo.declare
        def _(x: int, *args) -> sum:
            return [x, *args]

    def test_declare_return_post_processing(self):

        foo = Dispatch(identity)

        @foo.declare
        def int_to_list(x: int) -> list:
            return range(x)

        @foo.declare
        def str_to_int(
                x: str
        ) -> lambda x: reduce(lambda a, b: a + ord(b), x, 0):
            return x

        self.assertEqual(foo(5), list(range(5)))
        self.assertEqual(foo('hey'), 326)

        @foo.declare
        def split_dstring(d: dict, k: foo.Just('key')) -> (
        lambda x: x.split('_')):
            return d[k]

        self.assertEqual(foo({'key': 'hey_there'}, 'key'), ['hey', 'there'])


if __name__ == '__main__':
    unittest.main()
