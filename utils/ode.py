""" Numerical solutions to ODEs """

__author__ = "Frank Liu"
__version__ = 1.0

def runge_kutta(f):
    """ RK4 method for numerical integration
    RK4 solves \int_t^{t+dt} f(t, y) dt given f, t, and y.

    :param f: a function of y' = f(t,y)

    :return:
    a function of dy(t, y, dt) = \int_t^{t+dt} f(t, y) dt
    """
    return lambda t, y, dt: \
        (lambda dy1:
         (lambda dy2: 
           (lambda dy3:
            (lambda dy4:
             (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )(dt * f(t + dt, y + dy3))
           )(dt * f(t + dt/2, y + dy2/2))
         )(dt * f(t + dt/2, y + dy1/2))
        )(dt * f(t, y))
