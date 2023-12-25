# import pyplot
import matplotlib.pyplot as plt

# import casadi
from casadi import *

# import os
import os

def closedLoopPlots(data,plot_params,saveFig=False,figName=''):

    # create plot directory directory
    if not os.path.exists('Figures'): 
        os.makedirs('Figures') 

    # save colors
    violet = (0.4940, 0.1840, 0.5560)
    blue = (0, 0.4470, 0.7410)
    orange = (0.8500, 0.3250, 0.0980)
    yellow = (0.9290, 0.6940, 0.1250)
    red = (0.6350, 0.0780, 0.1840)
    green = (0.4660, 0.6740, 0.1880)
    lblue = (0.3010, 0.7450, 0.9330)

    # plotting range (x)
    x_range = plot_params['x_range']
    
    # extract plotting parameters
    iter_len = plot_params['iter_len']
    iter_step = plot_params['iter_step']
    time_len = plot_params['time_len']
    time_step = plot_params['time_step']

    # plot constraints
    plot_constraints = plot_params['plot_constraints']

    # font options    
    font_options = {'font.family': 'Times New Roman', 'mathtext.fontset' : 'cm',
                    'font.size': 11, 'axes.labelsize': 14}
    plt.rcParams.update(font_options)

    # extract optimal output
    x1_opt = data['x1_opt']
    x2_opt = data['x2_opt']
    u_opt = data['u_opt']

    # get state / input trajectory on first iteration
    X0 = data['X0']
    U0 = data['U0']

    # extract state entries
    X0_1 = X0[::2]
    X0_2 = X0[1::2]

    # get state / input trajectory on last iteration
    XM = data['XM']
    UM = data['UM']

    # extract state entries
    XM_1 = XM[::2]
    XM_2 = XM[1::2]

    # extract state and input constraints
    cst = data['cst']
    u_min = cst['u_min']
    u_max = cst['u_max']
    x_min = cst['x_min']
    x_max = cst['x_max']

    # plot 1: initial versus final state / input trajectory

    # create figure
    plt.figure()

    # get range of time-steps
    t_range = linspace(0,time_len,int(floor(time_len/time_step)))

    # plot state 1
    plt.subplot(311)
    # best controller
    plt.plot(t_range,x1_opt[:time_len:time_step],'-',marker='*',color=blue)
    # first iteration
    plt.plot(t_range,X0_1[:time_len:time_step],linestyle='dashdot',marker='s',markersize=4,color=violet)
    # last iteration
    plt.plot(t_range,XM_1[:time_len:time_step],'--',marker='.',color=orange)
    plt.ylabel('State $x_1$')
    plt.legend(['Best controller','Iteration $0$', 'Iteration $' + str(iter_len) + '$'])
    plt.tick_params(bottom = False, labelbottom = False) 
    plt.grid(True)
    plt.xlim(x_range)

    # plot state 2
    plt.subplot(312)
    # best controller
    plt.plot(t_range,x2_opt[:time_len:time_step],'-',marker='*',color=blue)
    # first iteration
    plt.plot(t_range,X0_2[:time_len:time_step],linestyle='dashdot',marker='s',markersize=4,color=violet)
    # last iteration
    plt.plot(t_range,XM_2[:time_len:time_step],'--',marker='.',color=orange)
    if plot_constraints:
        plt.plot(x_range,DM([x_min[1],x_min[1]]),color=red,linestyle='--')
    plt.ylabel('State $x_2$')
    plt.tick_params(bottom = False, labelbottom = False) 
    plt.grid(True)
    plt.xlim(x_range)

    # plot input
    plt.subplot(313)
    # best controller
    plt.plot(t_range,u_opt[:time_len:time_step],'-',marker='*',color=blue)
    # first iteration
    plt.plot(t_range,U0[:time_len:time_step],linestyle='dashdot',marker='s',markersize=4,color=violet)
    # last iteration
    plt.plot(t_range,UM[:time_len:time_step],'--',marker='.',color=orange)
    # constraints
    plt.plot(x_range,[u_min,u_min],color=red,linestyle='--')
    plt.plot(x_range,[u_max,u_max],color=red,linestyle='--')
    plt.ylabel('Input $u$')
    plt.xlabel('Time step $t$')
    plt.grid(True)
    plt.xlim(x_range)

    # save figure
    if saveFig:
        plt.savefig('Figures/' + figName + '_time_plot.pdf')

def iterPlot(gd,gd_2,plot_params,saveFig=False,figName=''):

    # create plot directory directory
    if not os.path.exists('Figures'): 
        os.makedirs('Figures') 

    # get costs
    cost_gd = gd['cost']
    # cost_gn = gn['cost']
    cost_gd_2 = gd_2['cost']
    opt_cost = gd['opt_cost']

    # get hyperparameters
    # rho_gn = gn['rho']
    rho_gd = gd['rho']
    eta_gd = gd['eta']
    rho_gd_2 = gd_2['rho']
    eta_gd_2 = gd_2['eta']

    # save colors
    orange = (0.8500, 0.3250, 0.0980)
    red = (0.6350, 0.0780, 0.1840)

    # extract plotting parameters
    iter_len = plot_params['iter_len']
    iter_step = plot_params['iter_step']

    # font options
    font_options = {'font.family': 'Times New Roman', 'mathtext.fontset' : 'cm',
                    'font.size': 12, 'axes.labelsize': 14}
    plt.rcParams.update(font_options)

    # get normalized cost
    normalized_cost_gd = (DM(cost_gd)-opt_cost)
    normalized_cost_gd_2 = (DM(cost_gd_2)-opt_cost)

    # get range of iterations
    iter_range = linspace(0,iter_len,int(floor(iter_len/iter_step)))

    # plot
    plt.figure(figsize=(6.4,3.2))
    plt.plot(iter_range,normalized_cost_gd[:iter_len:iter_step]/norm_2(opt_cost),'--',color=orange)
    plt.plot(iter_range,normalized_cost_gd_2[:iter_len:iter_step]/norm_2(opt_cost),linestyle='dashdot',color=red)
    plt.ylabel(r'$\frac{\mathcal{C}_k-\mathcal{C}^*}{\|\mathcal{C}^*\|}$')
    plt.xlabel('Iteration number')
    str1 = 'Gradient descent (' + '$\\rho=' + str(rho_gd) + '$, ' + '$\eta=' + str(eta_gd) + '$)'
    str2 = 'Gradient descent (' + '$\\rho=' + str(rho_gd_2) + '$, ' + '$\eta=' + str(eta_gd_2) + '$)'
    # str2 = 'Gauss-Newton (' + '$\\rho=' + str(rho_gn) + '$)'
    plt.legend([ str1, str2])
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()

    # save figure
    if saveFig:
        plt.savefig('Figures/' + figName + '_iter_plot.pdf')