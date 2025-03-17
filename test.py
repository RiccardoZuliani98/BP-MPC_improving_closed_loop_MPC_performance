from casadi import *

a = MX.sym('a',1,1)
# a = DM(1)

MSX = type(a)

B = MSX.eye(2)

print(B.shape)

