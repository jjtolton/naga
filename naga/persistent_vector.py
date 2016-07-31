import math
import pprint
import string
from functools import partial, reduce

import toolz


def rounded_log(n, base=2):
    log_dec = math.log(n, base)

    if int(log_dec) == log_dec:
        res = int(log_dec)
    else:
        res = int(log_dec) + 1
    return res


def iterate(f, x):
    res = x
    while True:
        yield res
        res = f(x)


def base_pv_insert(mask, shift, bits, base_vector, idx, item):
    path = pv_path(bits, idx, mask, shift)
    r1 = toolz.assoc_in(base_vector, path, item)
    len = base_vector['len']
    res = assoc(r1, 'len', len + 1)
    return res


def pv_path(bits, idx, mask, shift):
    path = ['root'] + [((idx >> (shift - i)) & mask) if (shift - i) > 0 else (idx & mask) for i in
                       range(0, shift + bits, bits)]
    return path


def list2vector(lst, bits=2):
    depth = rounded_log(len(lst), base=2 ** bits)
    width = 1 << bits
    mask = width - 1
    shift = bits * (depth - 1)

    base_vector = {'len': 0,
                   'root': {},
                   'bits': bits}

    pv_insert = partial(base_pv_insert, mask, shift, bits)

    res = reduce(lambda vector, _: apply(lambda n, item: pv_insert(vector, n, item), _), enumerate(lst), base_vector)

    return res


def iter_pv(pv):
    idx = 0
    bits = pv_bits(pv)
    depth = pv_depth(bits, pv)
    width = pv_width(bits)
    mask = pv_mask(width)
    shift = pv_shift(bits, depth)
    lenpv = pv_len(pv)

    while True:
        if idx == lenpv:
            raise StopIteration
        path = pv_path(bits, idx, mask, shift)
        idx += 1
        if len(path) < 2:
            res = toolz.get_in(path + [0], pv)
            yield res
        else:
            res = toolz.get_in(path, pv)
            yield res


def pv_bits(pv):
    bits = pv['bits']
    return bits


def pv_shift(bits, depth):
    shift = bits * (depth - 1)
    return shift


def pv_mask(width):
    mask = width - 1
    return mask


def pv_width(bits):
    width = 1 << bits
    return width


def pv_depth(bits, pv):
    depth = rounded_log(pv['len'], base=2 ** bits) if pv['len'] else 1
    return depth


def pv_len(pv):
    return pv['len']


# def pv_append(pv, item):
#     length = pv_len(pv)
#     res = pv_insert(pv, length, item)
#     return res


def pv_append(pv, item, new=False):
    bits = pv_bits(pv)
    depth = pv_depth(bits, pv) or 1
    width = pv_width(bits)
    mask = pv_mask(width)
    shift = pv_shift(bits, depth)
    idx = pv_len(pv)
    path = pv_path(bits, idx, mask, shift)

    if idx < (width ** depth):
        new_root = toolz.assoc(pv, 'len', idx + 1 if not new else idx)
        if new:
            res = toolz.assoc_in(new_root, path[:-1] + [path[-1] - 1], item)
        else:
            res = toolz.assoc_in(new_root, path, item)
        return res

    # no room in node, no room in root
    else:
        new_root = toolz.assoc(toolz.assoc(new_pv(pv['root'], bits), 'len', idx + 1), 'root', {0: pv['root'],
                                                                                               1: {}})
        res = pv_append(new_root, item, new=True)

    return res


def new_pv(item=None, bits=5):
    return {'tail': [],
            'root': {0: item} if item else {},
            'len': 1 if item else 0,
            'bits': bits}


def binseq2int(direction, binseq):
    return sum(map(lambda x: 2 ** x[0] if x[1] else 0, enumerate(binseq[direction])))


def lbinseq2int(binseq):
    return sum(map(lambda x: 2 ** x[0] if x[1] else 0, enumerate(binseq[::-1])))

def bbinseq2int(binseq):
    return sum(map(lambda x: 2 ** x[0] if x[1] else 0, enumerate(binseq)))

def shiftidx(n):
    c = 0
    while n > 0:
        c += 1
        n >>= 1
    return c


def calc_upper_subpath(pv, idx):
    path = calculate_path(idx, pv)
    nonroot = path[1:]
    root = path[:1]
    subpath = root + nonroot[:len(nonroot) - shiftidx(lbinseq2int(nonroot))]
    return subpath

def calc_lower_subpath(pv, idx):
    path = calculate_path(idx, pv)[1:]
    subpath = path[:len(path) - shiftidx(lbinseq2int(path))]
    return subpath

def pv_insert(pv, idx, item):
    path = calculate_path(idx, pv)
    val = toolz.get_in(path[:-1], pv)
    new_val = toolz.assoc(val, len(val), item)
    res = toolz.assoc_in(pv, path[:-1], new_val)
    return res


def calculate_path(idx, pv):
    bits = pv_bits(pv)
    depth = pv_depth(bits, pv)
    width = pv_width(bits)
    mask = pv_mask(width)
    shift = pv_shift(bits, depth)
    path = pv_path(bits, idx, mask, shift)
    return path


def main():
    lst = list(string.ascii_lowercase)
    vec = new_pv(bits=1)
    for i in range(10):
        vec = pv_append(vec, i)
        # pprint.pprint(vec)
        # print(list(iter_pv(vec)))
    # pprint.pprint(vec)
    print(lbinseq2int([0, 0, 1, 1, 1]))
    # print(calc_lower_subpath(vec, 3))
    print(calc_upper_subpath(vec, 3))
    pprint.pprint(toolz.get_in(calc_upper_subpath(vec, 3), vec))


if __name__ == '__main__':
    main()
