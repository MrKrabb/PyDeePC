# To run this example you also need to install matplotlib
import numpy as np
import math
import scipy.signal as scipysig
import cvxpy as cp
from cvxpy import Expression, Variable, Problem, Parameter
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, Callable, List, Optional, Union, Dict



from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc.utils import Data
from utils import System

class OptimizationProblemVariables(NamedTuple):
    """
    Class used to store all the variables used in the optimization
    problem
    """
    u_ini: Union[Variable, Parameter]
    y_ini: Union[Variable, Parameter]
    s: Union[Variable, Parameter]
    u: Union[Variable, Parameter]
    y: Union[Variable, Parameter]
    g: Union[Variable, Parameter]
    slack_y: Union[Variable, Parameter]
    slack_u: Union[Variable, Parameter]


class OptimizationProblem(NamedTuple):
    """
    Class used to store the elements an optimization problem
    :param problem_variables:   variables of the opt. problem
    :param constraints:         constraints of the problem
    :param objective_function:  objective function
    :param problem:             optimization problem object
    """
    variables: OptimizationProblemVariables
    constraints: List[Constraint]
    objective_function: Expression
    problem: Problem

class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param y: output data
    """
    u: np.ndarray
    y: np.ndarray


def create_hankel_matrix(data: np.ndarray, order: int) -> np.ndarray:
    """
    Create an Hankel matrix of order L from a given matrix of size TxM,
    where M is the number of features and T is the batch size.
    Note that we need L <= T.

    :param data:    A matrix of data (size TxM). 
                    T is the batch size and M is the number of features
    :param order:   the order of the Hankel matrix (L)
    :return:        The Hankel matrix of type np.ndarray
    """
    data = np.array(data)
    
    assert len(data.shape) == 2, "Data needs to be a matrix"

    T,M = data.shape
    assert T >= order and order > 0, "The number of data points needs to be larger than the order"

    H = np.zeros((order * M, (T - order + 1)))
    for idx in range (T - order + 1):
        H[:, idx] = data[idx:idx+order, :].flatten()
    return H

def split_data(data: Data, Tini: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Utility function used to split the data into past data and future data.
    Constructs the Hankel matrix for the input/output data, and uses the first
    Tini rows to create the past data, and the last 'horizon' rows to create
    the future data.
    For more info check eq. (4) in https://arxiv.org/pdf/1811.05890.pdf

    :param data:    A tuple of input/output data. Data should have shape TxM
                    where T is the batch size and M is the number of features
    :param Tini:    number of samples needed to estimate initial conditions
    :param horizon: horizon
    :return:        Returns Up,Uf,Yp,Yf (see eq. (4) of the original DeePC paper)
    """
    assert Tini >= 1, "Tini cannot be lower than 1"
    assert horizon >= 1, "Horizon cannot be lower than 1"

    Mu, My = data.u.shape[1], data.y.shape[1]
    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)

    Up, Uf = Hu[:Tini * Mu], Hu[-horizon * Mu:]
    Yp, Yf = Hy[:Tini * My], Hy[-horizon * My:]
    
    return Up, Uf, Yp, Yf

def low_rank_matrix_approximation(
        X: np.ndarray,
        explained_var: Optional[float] = 0.9,
        rank: Optional[int] = None,
        SVD: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        **svd_kwargs):
    """
    Computes an low-rank approximation of a matrix

    Adapted from https://gist.github.com/thearn/5424219

    :param X:               matrix to approximate
    :param explained_var:   Value in (0,1] used to compute the rank. Describes how close 
                            the low rank matrix is to the original matrix (in terms of the
                            singular values). Default value: 0.9
    :param rank:            rank order. To be used if you want to approximate the matrix by a specific
                            rank. By default is None. If different than None, then it will override the
                            explained_var parameter.
    :param SVD:             If not none, it should be the SVD decomposition (see numpy.linalg.svd) of X
    :param **svd_kwargs:    additional parameters to be passed to numpy.linalg.svd
    :return: the low rank approximation of X
    """
    assert len(X.shape) == 2, "X must be a matrix"
    assert explained_var is None and isinstance(rank, int) or isinstance(explained_var, float), \
        "You need to specify explained_var or rank!"
    assert explained_var is None or explained_var <= 1. and explained_var > 0, \
        "explained_var must be in (0,1]"
    assert rank is None or (rank >= 1 and rank <= min(X.shape[0], X.shape[1])), \
        "Rank cannot be lower than 1 or greater than min(num_rows, num_cols)"
    assert SVD is None or len(SVD) == 3, "SVD must be a tuple of 3 elements"

    u, s, v = np.linalg.svd(X, **svd_kwargs) if not SVD else SVD

    if rank is None:
        s_squared = np.power(s, 2)
        total_var = np.sum(s_squared)
        z = np.cumsum(s_squared) / total_var
        rank = np.argmax(np.logical_or(z > explained_var, np.isclose(z, explained_var)))

    X_low = np.zeros_like(X)

    for i in range(rank):
        X_low += s[i] * np.outer(u[:,i], v[i])
    return X_low





class DeePC(object):
    optimization_problem: OptimizationProblem = None
    _SMALL_NUMBER: float = 1e-32

    def __init__(self, data: Data, Tini: int, horizon: int):
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        :param Tini:                number of samples needed to estimate initial conditions
        :param horizon:             horizon length
        :param explained_variance:  Regularization term in (0,1] used to approximate the Hankel matrices.
                                    By default is None (no low-rank approximation is performed).
        """
        self.Tini = Tini
        self.horizon = horizon
        self.update_data(data)

        self.optimization_problem = None

    def update_data(self, data: Data):
        """
        Update Hankel matrices of DeePC. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], \
            "Input/output data must have the same length"
        assert data.y.shape[0] - self.Tini - self.horizon + 1 >= 1, \
            f"There is not enough data: this value {data.y.shape[0] - self.Tini - self.horizon + 1} needs to be >= 1"
        
        Up, Uf, Yp, Yf = split_data(data, self.Tini, self.horizon)

        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        
        self.M = data.u.shape[1]
        self.P = data.y.shape[1]
        self.T = data.u.shape[0]

        self.optimization_problem = None

    def build_problem(self,
            build_loss: Callable[[cp.Variable, cp.Variable, cp.Parameter], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable, cp.Parameter], Optional[List[Constraint]]]] = None,
            lambda_g: float = 0.,
            lambda_y: float = 0.,
            lambda_u: float= 0.,
            lambda_proj: float = 0.) -> OptimizationProblem:
        """
        Builds the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        For info on the projection (least-square) regularizer, see also
        https://arxiv.org/pdf/2101.01273.pdf


        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :param lambda_g:            non-negative scalar. Regularization factor for g. Used for
                                    stochastic/non-linear systems.
        :param lambda_y:            non-negative scalar. Regularization factor for y_init. Used for
                                    stochastic/non-linear systems.
        :param lambda_u:            non-negative scalar. Regularization factor for u_init. Used for
                                    stochastic/non-linear systems.
        :param lambda_proj:         Positive term that penalizes the least square solution.
        :return:                    Parameters of the optimization problem
        """
        assert build_loss is not None, "Loss function callback cannot be none"
        assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
        assert lambda_u >= 0, "Regularizer of u_init must be non-negative"
        assert lambda_proj >= 0, "The projection regularizer must be non-negative"

        self.optimization_problem = False

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.Tini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.Tini), name='y_ini')
        s = cp.Parameter(shape=(self.P * self.horizon), name='s')
        u = cp.Variable(shape=(self.M * self.horizon), name='u')
        y = cp.Variable(shape=(self.P * self.horizon), name='y')
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1), name='g')
        slack_y = cp.Variable(shape=(self.Tini * self.P), name='slack_y')
        slack_u = cp.Variable(shape=(self.Tini * self.M), name='slack_u')

        Up, Yp, Uf, Yf = self.Up, self.Yp, self.Uf, self.Yf

        if lambda_proj > DeePC._SMALL_NUMBER:
            # Compute projection matrix (for the least square solution)
            Zp = np.vstack([Up, Yp, Uf])
            ZpInv = np.linalg.pinv(Zp)
            I = np.eye(self.T - self.Tini - self.horizon + 1)
            # Kernel orthogonal projector
            I_min_P = I - (ZpInv@ Zp)

        A = np.vstack([Up, Yp, Uf, Yf])
        b = cp.hstack([uini + slack_u, yini + slack_y, u, y])

        # Build constraints
        constraints = [A @ g == b]

        if math.isclose(lambda_y, 0):
            constraints.append(cp.norm(slack_y, 2) <= DeePC._SMALL_NUMBER)
        if math.isclose(lambda_u, 0):
            constraints.append(cp.norm(slack_u, 2) <= DeePC._SMALL_NUMBER)

        # u, y = self.Uf @ g, self.Yf @ g
        u = cp.reshape(u, (self.horizon, self.M))
        y = cp.reshape(y, (self.horizon, self.P))

        _constraints = build_constraints(u, y, s) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y, s)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        # Add regularizers
        _regularizers = lambda_g * cp.norm(g, p=1) if lambda_g > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_y * cp.norm(slack_y, p=1) if lambda_y > DeePC._SMALL_NUMBER else 0
        _regularizers += lambda_proj * cp.norm(I_min_P @ g) if lambda_proj > DeePC._SMALL_NUMBER  else 0
        _regularizers += lambda_u * cp.norm(slack_u, p=1) if lambda_u > DeePC._SMALL_NUMBER else 0

        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(
                u_ini = uini, y_ini = yini, s = s, u = u, y = y, g = g, slack_y = slack_y, slack_u = slack_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def solve(
            self,
            data_ini: Data,
            s,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data_ini:            A tuple of input/output data used to estimate initial condition.
                                    Data should have shape Tini x M where Tini is the batch size and
                                    M is the number of features
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(data_ini.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data_ini.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data_ini.u.shape[1] == self.M, "Incorrect number of features for the input signal"
        assert data_ini.y.shape[1] == self.P, "Incorrect number of features for the output signal"
        assert data_ini.y.shape[0] == data_ini.u.shape[0], "Input/output data must have the same length"
        assert data_ini.y.shape[0] == self.Tini, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        # Need to transpose to make sure that time is over the columns, and features over the rows
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        self.optimization_problem.variables.u_ini.value = uini
        self.optimization_problem.variables.y_ini.value = yini

        self.optimization_problem.variables.s.value = s

        try:
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        u_optimal = (self.Uf @ self.optimization_problem.variables.g.value).reshape(self.horizon, self.M)
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info


# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable, s:cp.Parameter) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    s=s[:,np.newaxis]
    return  cp.norm(y-s,'fro')**2

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable, s:cp.Parameter) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    # no real constraints on y, input should be between -1 and 1
    return [u >= -1, u <= 1]

# DeePC paramters
s = 1                       # How many steps before we solve again the DeePC problem
T_INI = 2                   # Size of the initial set of data
T_list = [100]              # Number of data points used to estimate the system
HORIZON = 10                # Horizon length
LAMBDA_G_REGULARIZER = 0    # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0    # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0    # u regularizer
EXPERIMENT_HORIZON = 100    # Total number of steps

# model of two-tank example
A = np.array([
        [0.70469, 0.     ],
        [0.24664, 0.70469]])
B = np.array([[0.75937], [0.12515]])
C = np.array([[0., 1.]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1))

fig, ax = plt.subplots(1,2)
plt.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f'Simulating with {T} initial samples...')
    sys.reset()
    # Generate initial data and initialize DeePC
    
    data = sys.apply_input(u = np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
    sys.reset(data_ini = data_ini)

    deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = LAMBDA_G_REGULARIZER,
        lambda_y = LAMBDA_Y_REGULARIZER,
        lambda_u = LAMBDA_U_REGULARIZER)

    for step in range(EXPERIMENT_HORIZON//s):
        # Solve DeePC
        time= step * s + np.arange(0, HORIZON) * s
        ref = 1+0.1*np.sin(time)
        path = np.zeros((HORIZON))
        u_optimal, info = deepc.solve(data_ini = data_ini, s=ref, warm_start=True)

        # Apply optimal control input
        _ = sys.apply_input(u = u_optimal[:s, :], noise_std=1e-2)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')
    ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')

ax[0].set_ylim(0, 1.5)
ax[1].set_ylim(-1.2, 1.2)
ax[0].set_xlabel('t')
ax[0].set_ylabel('y')
ax[0].grid()
ax[1].set_ylabel('u')
ax[1].set_xlabel('t')
ax[1].grid()
ax[0].set_title('Closed loop - output signal $y_t$')
ax[1].set_title('Closed loop - control signal $u_t$')
plt.legend(fancybox=True, shadow=True)
plt.show()