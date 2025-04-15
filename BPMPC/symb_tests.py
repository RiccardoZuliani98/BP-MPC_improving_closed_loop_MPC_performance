import casadi as ca
from symb import Symb
import timeit

sym1 = Symb()

x = ca.SX.sym('x',2,1)
x0 = ca.DM.ones(2,1)

sym1.addVar('x',x,x0)

sym2 = Symb()

y = ca.SX.sym('y',2,1)
y0 = ca.DM.ones(2,1)

sym2.addVar('y',y,y0)

sym1 += sym2

print(sym1.getVar('y'))
print(sym1.var)


print(timeit.timeit('(sym1+sym2).getVar(\'y\')',number=2000, globals=globals())/2000)
print(timeit.timeit('(sym1.add2(sym2)).getVar(\'y\')',number=2000, globals=globals())/2000)