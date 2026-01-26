from odl.core.operator import Operator, IdentityOperator
from odl.core.space import ProductSpace
from typing import Callable


class ODEStep(Operator):
    """
    Operator representing the step S of an ODE Solver, relevant to track the backpropagation.

    The idea is to offer step-based structure to solving dy/dt = f(y, q, t), so in this case
    this operator is S, such that the solver is y^{n+1} = S(q, y^{n}, t_n)
    """
    def __init__(self, y_domain, q_domain, f, dt, t):
        assert(f.domain) == ProductSpace(q_domain, y_domain)
        super().__init__(domain=f.domain, range=y_domain)

    def derivative(self, q, y):
        """
        Returns:
            Sq : LinearOperator Q -> Y
            Sy : LinearOperator Y -> Y

            Where Sq is the partial derivative of the step operator S with respect to the control q
            and Sy is the partial derivative of the step operator with respect to the state y.
        """
        raise NotImplementedError

    def derivative(self, q_y):
        raise NotImplementedError
    
class ODESolverDerivative(Operator):
    def __init__(self, step, qt, states):
        """
        Docstring for __init__
        
        :param self: Description
        :param step: Description
        :param qt: Description
        :param states: Description
        """
        self.states = states
        self.qt = qt
        self.step = step
        super().__init__(domain=qt.space, range=states[0].space)

    def _call(self, dqt):
        dy = self.states[0].space.zero()
        for i in range(self.N):
            Sq, Sy = self.step.derivative(self.qt[i], self.states[i])
            dy = Sy(dy) + Sq(dqt[i])
        return dy

    def adjoint(self, eta):
        lam = eta
        grads = [None] * self.N

        for i in reversed(range(self.N)):
            Sq, Sy = self.step.derivative(self.qt[i], self.states[i])

            # gradient contribution
            grads[i] = Sq.adjoint(lam)

            # propagate adjoint state
            lam = Sy.adjoint(lam)

        return self.domain.element(grads)
    

class ODESolverFixedy0(Operator):
    def __init__(self, f, y0, Nt, domain, step: Callable[[],ODEStep], **kwargs):
        '''Iterative solver on the following first order ODE: dy/dt = f(q, y, t); y(0) = y0.

            Since all of the most relevant ODE solvers are iterative based this allows -
            - a general structure to solve several first order ODE, with different control v -
            - and with free choice for step that allow general numerical analysis choices.  
            
            f: represents the equation itself.
            q: represents the coefficients/parameters of the equation.
            y0: initial condition.

            Solvers can be either functions of the coefficients v or -
            - or either function of the initial condition.

            In this case this operator is defined to be a function of the coefficients -
            - of the equation, also known as the control problem.

            ¡WARNING! The derivative implemented for this operator is unfortunately
            way too expensive computationally respect to using autograd with torch.
        '''
        self._f = f
        self.N = Nt
        self.y0 = y0
        self.y_space = self.y0.space
        self.dt = 1.0 / Nt
        self.q_space = domain
        self.step = step
        super(ODESolverFixedy0, self).__init__(domain=ProductSpace(self.q_space, Nt, weighting=1/Nt), range=y0.space)

    def _call(self, qt):
        # operator = IdentityOperator(self.y_space)
        step_domain = ProductSpace(self.q_space, self.y_space)
        y = self.y0
        for i in range(self.N):
#            operator = operator @ self.step(self._f, vt[i], self.dt, t=i*self.dt)
            y = self.step(self._f, self.dt, t=i*self.dt)(step_domain.element(qt[i], y))
            # operator = operator @ self.step(self.F, self.dt, t=i*self.dt)

        return y
#        return operator(odl.ProductSpace(...).element(self.y0, vt[0]))

    def state_generator(self, qt):
        # operator = IdentityOperator(self.y_space)
        step_domain = ProductSpace(self.q_space, self.y_space)
        ys = []
        y = self.y0
        ys.append(y)
        for i in range(self.N):
#            operator = operator @ self.step(self._f, vt[i], self.dt, t=i*self.dt)
            y = self.step(self._f, self.dt, t=i*self.dt)(step_domain.element(qt[i], y))
            ys.append(y)
            # operator = operator @ self.step(self.F, self.dt, t=i*self.dt)

        return ys

    def derivative(self, qt, ):
        ys = self.state_generator(qt)
        
        return ODESolverDerivative(self.step, qt, ys)
    
    def adjoint(self, point):
        raise NotImplementedError
    
class EulerStep(ODEStep):
    def __init__(self, y_domain, q_domain, f, dt, t):
        """
        Explicit Euler iteration: y^{i+1} = y^{i} + F(q_i, y^{i}, t_i)*dt

        

        :param y_domain: Description
        :param q_domain: Description
        :param f: Description
        :param dt: Description
        :param t: Description
        """
        self.f = f
        self.dt = dt
        self.t = t
        super().__init__(y_domain, q_domain, f, dt, t)

    def _call(self, q_y):
        q, y = q_y
        return y + self.dt * self.f(q, y, self.t)

    def derivative(self, q_y):
        q, y = q_y
        Df_q, Df_y = self.f.derivative(q, y, self.t)

        Sq = self.dt * Df_q      # Q → Y
        Sy = IdentityOperator(y.space) + self.dt * Df_y

        return Sq, Sy
        

    
class RK4Step(ODEStep):
    def __init__(self, y_domain, q_domain, f, dt, t):
        self.f = f
        self.dt = dt
        self.t = t
        super().__init__(y_domain, q_domain, f, dt, t)

    def _call(self, q_y):
        q, y = q_y
        k1 = self.f(q, y, self.t)
        k2 = self.f(q, y + 0.5 * self.dt * k1, self.t + 0.5)
        k3 = self.f(q, y + 0.5 * self.dt * k2, self.t + 0.5)
        k4 = self.f(q, y + self.dt * k3, self.t + 1.0)

        return y + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def derivative(self, q_y):
        raise NotImplementedError


class ODESolverFixedq(Operator):
    def __init__(self, f, v, Nt, domain=None, **kwargs):
        raise NotImplementedError
    
    def _call(self, y0):
        raise NotImplementedError






