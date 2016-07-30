import unittest

from python_code.functional_tools.tools import apply, merge, assoc, dissoc, merge_with, merge_with_default, assoc_in, \
    update_in, terminal_dict


class FuncyToolsTest(unittest.TestCase):
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
        expected_res = {(1, 2), (3, 4), (4, 5), ('a', 'b'), ('c', 'd'), ('e', 'f'), ('name', 'taco')}
        res = set(apply(*ds))
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
        res = merge_with(lambda x, y: x + y if (isinstance(x, int) and isinstance(y, int)) else y, a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_doubles_strings_only(self):
        a, _, c = self.ds
        expected_res = {1: 2, 3: 4, 4: 5, 'name': 'tacotaco'}
        res = merge_with(lambda x, y: x + y if isinstance(x, str) else y,
                         a, a, c, c)
        self.assertEqual(expected_res, res)

    def test_merge_with_default_cats_to_list(self):
        ds = self.ds
        expected_res = {1: [2, 2], 3: [4, 4], 4: [5, 5], 'a': ['b', 'b'], 'c': ['d', 'd'], 'e': ['f', 'f'],
                        'name': ['taco', 'taco']}
        res = merge_with_default(lambda x, y: x + [y], [], *(ds + ds))
        self.assertEqual(expected_res, res)

    def test_merge_with_default_sums_or_adds(self):
        a, *_ = self.ds
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

        res = assoc_in(d0, ['a', 'd', 'f'], {'1': '2', '3': 'hello'})
        self.assertEqual(expected_res, res)

    def test_update_in_destructively_adds_key_to_path(self):
        _, _, c = self.ds
        initial_d = assoc_in(c, ['thisis'], {'a': dict})
        expected_res = {'name': 'taco',
                        'thisis': {'b': dict}}
        res = update_in(initial_d, ['thisis', 'b'], dict)
        self.assertEqual(expected_res, res)

    def test_terminal_dicts_id_terminal_dicts_correctly(self):
        a = {'a': ['this', 'is', 'terminal']}
        b = {'b': {'this': 'ain\'t'}}
        self.assertTrue(terminal_dict(a))
        self.assertFalse(terminal_dict(b))
