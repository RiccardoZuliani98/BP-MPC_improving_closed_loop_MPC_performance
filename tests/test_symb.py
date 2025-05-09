import sys
import os
import casadi as ca

# add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.symb import Symb


sym1 = Symb()

x = ca.SX.sym('x',2,1)
x0 = ca.DM.ones(2,1)

sym1.addVar('x',x,x0)

sym2 = Symb()

y = ca.SX.sym('y',2,1)
y0 = ca.DM.ones(2,1)

sym2.addVar('y',y,y0)

sym1 += sym2

print(sym1.get_var('y'))
print(sym1.var)


sym1+sym2.getVar('y')
(sym1.add2(sym2)).getVar('y')