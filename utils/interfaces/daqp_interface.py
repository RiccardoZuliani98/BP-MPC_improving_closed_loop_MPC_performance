import daqp
import numpy as np
from ctypes import *
from utils.interfaces.callable_wrapper import CallableWrapper

def daqp_interface(is_equality):

    # convert true => 1 and false => 2
    is_equality_converted = np.array(np.where(is_equality, 1, 2),dtype=c_int)

    def call_solver(qp_ingredients):

        H = np.array(qp_ingredients['h'],dtype=c_double)
        f = np.array(qp_ingredients['g'],dtype=c_double).squeeze()
        A = np.array(qp_ingredients['a'],dtype=c_double)
        bl = np.array(qp_ingredients['lba'],dtype=c_double).squeeze()
        bu = np.array(qp_ingredients['uba'],dtype=c_double).squeeze()

        x_star,_,exitflag,info = daqp.solve(H,f,A,bu,bl,is_equality_converted)

        assert exitflag in [1,2], 'QP not solved correctly'

        return {'x':x_star,'lam_a':info['lam']}

    return CallableWrapper(call_solver)