import casadi as ca
import numpy as np

def rls(dynamics,horizon:int,lam:float,theta0:ca.DM=None,jit:bool=False):

    # check if dynamics should be compiled
    if jit:
        jit_options = {"flags": "-O3", "verbose": False, "compiler": "gcc -Ofast -march=native"}
        compilation_options = {"jit": True, "compiler": "shell", "jit_options": jit_options}
    else:
        compilation_options = {}

    # check that nominal model is not fully known
    assert 'theta' in dynamics.param_nom, 'Theta should be set as nominal parameter in dynamics.'

    # check that theta is initialized
    if theta0 is None:
        assert 'theta' in dynamics.init, 'Theta must be initialized.'
        theta0 = dynamics.init['theta']

    # extract parameters
    theta,x,u = dynamics.param_nom['theta'],dynamics.param_nom['x'],dynamics.param_nom['u']

    # represent the model as f(x,u) = theta.T@phi(x,u)
    phi_sym = ca.jacobian(dynamics.x_next_nom,theta)

    # check that jacobian does not depend on theta
    assert not ca.depends_on(phi_sym,theta), 'Model is not parameter affine.'

    # turn into function
    phi_single = ca.Function('phi',[x,u],[phi_sym])

    # map to accept entire trajectories
    phi = phi_single.map(horizon,[False,False],[False],compilation_options)

    # precompute dimension of theta
    n_theta = theta.shape[0]

    def sys_id_update(sim,running_vars,k):

        # get past a and 
        a_k = running_vars['A']
        b_k = running_vars['b']

        # compute feature vectors
        phi_k = np.array(phi(sim.x[:,:-1],sim.u))

        # compute output vector
        z_k = np.array(sim.x[:,1:])

        # reshape to (horizon, *phi.shape)
        phi_reshaped = phi_k.reshape(phi_k.shape[0],-1,horizon,order='F').transpose(2,1,0)

        # update a and b
        a_k_1 = ca.DM(a_k + np.einsum('nij,njk->ik', phi_reshaped, phi_reshaped.transpose(0,2,1)))
        b_k_1 = ca.DM(b_k + np.atleast_2d(np.einsum('nij,nj->i', phi_reshaped, z_k.T)).T)

        # compute new model
        theta = ca.solve(a_k_1,b_k_1)

        # # test against for loop
        # phi_k_list = np.split(phi_k.T,horizon,axis=0)
        # z_k_list = np.split(z_k,horizon,axis=1)

        # # preallocate outer products
        # product_1 = np.zeros((phi_k_list[0].shape[0],phi_k_list[0].shape[0]))
        # product_2 = np.zeros((phi_k_list[0].shape[0],z_k_list[0].shape[1]))

        # # test against for loop
        # for i in range(horizon):
        #     product_1 = product_1 + phi_k_list[i]@(phi_k_list[i].T)
        #     product_2 = product_2 + phi_k_list[i]@z_k_list[i]

        # # check that this is equal to the list above
        # for idx, elem in enumerate(phi_k_list):
        #     assert np.allclose(elem,phi_reshaped[idx])

        # run through the horizon and perform the RLS updates
        new_psi = {'A':a_k_1,'b':b_k_1,'theta':theta}

        return sim.psi | new_psi
    
    def sys_id_init():
        return {'A':ca.DM.eye(n_theta)*lam,'b':theta0}
    
    return sys_id_update, sys_id_init