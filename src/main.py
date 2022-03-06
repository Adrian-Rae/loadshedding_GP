from GPAtom import Terminal, Operator
from GPPopulator import Populator

if __name__ == '__main__':
    t_set = [Terminal[int](), Terminal[int]()]
    o_set = [Operator[int](1), Operator[int](3)]
    p = Populator(5, 2, t_set, o_set)
    for k in p.generate():
        print(k)
