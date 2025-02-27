import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class animPendulum:

    def __init__(self, theta_vals,xcartpositons, ycart, pendlength, m ) -> None:

        def bob_xy(x_pos, theta_vals,ycart, l):
            x_val = l * np.sin(theta_vals) + x_pos
            y_val = l * np.cos(theta_vals) + ycart
            return x_val, y_val


        def _animate(i):
            time_text.set_text(f'frame = {i:.1f} s,  theta = {theta_vals[i]} \n, x = {xcartpositons[i]}')
            x_bob, y_bob = xbob[i], ybob[i]
            x_cart = xcartpositons[i]
            ln1.set_data([x_bob], [y_bob])
            rod.set_data([xcartpositons[i], x_bob], [ycart, y_bob])
            trolley.set_data([x_cart-1, x_cart+1], [ycart])
        


        fig, ax = plt.subplots(1,1)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        time_text.set_text('')
        ax.grid()
        ln1, = plt.plot([],[], 'ro', markersize = (m*5)**0.5)
        rod, = ax.plot([], [], 'k-', lw=2)  # black line for the rod
        trolley, = ax.plot([],[], 'k-', lw=2)
        ax.set_ylim(-1,16)
        ax.set_xlim(-20,20)
        ax.set_aspect('equal')
        xbob, ybob = bob_xy(xcartpositons, theta_vals,ycart, pendlength)

        
        ani = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(xcartpositons),
        interval = 100,
        repeat = True
        )

        plt.show()