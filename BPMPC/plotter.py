from casadi import *
import matplotlib.pyplot as plt

class Plotter:

    def __init__(self):
        pass

    @staticmethod
    def colors():
        # save colors
        violet = (0.4940, 0.1840, 0.5560)
        blue = (0, 0.4470, 0.7410)
        orange = (0.8500, 0.3250, 0.0980)
        yellow = (0.9290, 0.6940, 0.1250)
        red = (0.6350, 0.0780, 0.1840)
        green = (0.4660, 0.6740, 0.1880)
        lblue = (0.3010, 0.7450, 0.9330)

        return {'violet':violet,'blue':blue,'orange':orange,'yellow':yellow,'red':red,'green':green,'lblue':lblue}

    @staticmethod
    def plotTrajectory(S, options=None, show=False):

        if options is None:
            options = {}

        # extract options
        options = {'x':[0,1,2,3],'x_legend':['Position','Velocity','Angle','Angular velocity'],'u':[0],'u_legend':['Force'],'plot_constraints':True, 'color':'blue'} | options

        # extract colors
        colors = Plotter.colors()

        # extract state trajectory
        x = S.x_mat
        u = S.u_mat

        # get dimension
        T = x.shape[1] - 1

        # create time vector
        t = np.arange(T + 1)

        ### 1. STATE FIGURE

        if len(options['x']) > 0:
            # Check if figure 1 exists, if not create it
            if not plt.fignum_exists(1):
                # Create a new figure and subplots
                fig, axs = plt.subplots(len(options['x']), 1, num=1, figsize=(5, 5))
            else:
                # Use the existing figure and axes
                fig = plt.figure(1)
                axs = fig.get_axes()

            # Ensure axs is always a list (even if it's just one subplot)
            if not (isinstance(axs, np.ndarray) or isinstance(axs, list)):
                axs = [axs]

            # plot state trajectory
            for i in range(len(options['x'])):
                ax = axs[i]
                ax.plot(t, np.array(x[i, :]).squeeze(), label=options['x_legend'][i], color=colors[options['color']])
                ax.legend()
            axs[0].set_title('State trajectory')

            # adjust layout
            plt.tight_layout()

        ### 2. INPUT FIGURE

        if len(options['u']) > 0:
            # Check if figure 2 exists, if not create it
            if not plt.fignum_exists(2):
                # Create a new figure and subplots
                fig, axs = plt.subplots(len(options['u']), 1, num=2, figsize=(5, 5))
            else:
                # Use the existing figure and axes
                fig = plt.figure(2)
                axs = fig.get_axes()

            # Ensure axs is always a list (even if it's just one subplot)
            if not (isinstance(axs, np.ndarray) or isinstance(axs, list)):
                axs = [axs]

            # plot input trajectory
            for i in range(len(options['u'])):
                ax = axs[i]
                ax.plot(t[:-1], np.array(u[i, :]).squeeze(), label=options['u_legend'][i], color=colors[options['color']])
                ax.legend()
            axs[0].set_title('Input trajectory')

        # adjust layout
        plt.tight_layout()

        # update plot without blocking
        plt.draw()

        if show:
            plt.show()  # Only call plt.show() after plotting both trajectories