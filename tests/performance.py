from _operator import mul
from random import choice

from naga import Dispatch, reduce


@Dispatch
def fib(*args):
    raise TypeError("Unsupported type(s)")


@fib.declare
def fib_base_case(n: {0, 1}):
    return n


@fib.declare
def fib_recursive_case(n: int):
    return fib(n - 2) + fib(n - 1)


@fib.declare
def make_fib_list(a: {list}, b: int):
    return [fib(n) for n in range(b)]


@fib.declare
def fibsumlist(a: {sum}, b: int) -> sum:
    return fib(list, b)


@fib.declare
def fibmul(a: {mul}, b: int):
    return reduce(a, map(fib, range(1, b)))

def empty_if_none(x):
    if x is None:
        return []
    return x


@fib.declare
def fib(choice: {choice}) -> empty_if_none:
    return choice([None, 100])

@fib.declare
def fib(choice: {choice}, n: int) -> list:
    for i in range(n):
        yield fib(choice)

if __name__ == '__main__':
    print(fib_base_case(0))
    print(fib(0))
    print(fib(1))
    print(fib(10))
    print(fib_recursive_case(10))
    print(fib(list, 10))
    print(make_fib_list(list, 10))
    print(fib(sum, 10))
    print(fibsumlist(sum, 10))
    print(fib(mul, 10))
    print(fibmul(mul, 10))
    print(fib(choice, 10))
