#!/usr/bin/env python
""" Demonstrate how to simulate dynamics forwards and backwards in time using RK4.
"""

from utils.ode import runge_kutta
import numpy as np
import matplotlib.pyplot as plt

def main():
    f = lambda t, y: (np.cos(t) + np.sin(t)) * np.exp(t) 
    dy = runge_kutta(f)
    dt = 0.1
    time = np.linspace(0, 10, 100, endpoint=False)
    y_real = lambda t: np.sin(t) * np.exp(t)

    y = y_real(time[0])
    forward_sim = []
    for t in time:
        forward_sim.append((t, y))
        y = y + dy(t, y, dt)


    y = y_real(time[-1])
    backward_sim = []
    for t in reversed(time):
        backward_sim.append((t, y))
        y = y + dy(t, y, -dt)

    forward_sim = np.asarray(forward_sim)
    backward_sim = np.asarray(backward_sim)
    forward_sim_error = np.fabs(forward_sim[:,1] - y_real(time))
    backward_sim_error = np.fabs(np.flipud(backward_sim[:,1]) - y_real(time))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(forward_sim[:, 0], forward_sim[:, 1], 'r', label="Forward sim")
    plt.plot(backward_sim[:, 0], backward_sim[:, 1], 'xg', label="Backward sim")
    plt.xlabel("Time [s]")
    plt.ylabel("y(t)")
    plt.title("Solution of y' = (cos(t) + sin(t))*exp(t) given y(0) = {} and dt={}".format(y_real(0), dt))
    plt.grid()
    plt.legend()

    plt.subplot(212)
    plt.plot(time, forward_sim_error, 'r', label="forward sim error")
    plt.plot(time, backward_sim_error, 'xg', label="backward sim error")
    plt.xlabel("Time [s]")
    plt.ylabel("Sim error")
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
