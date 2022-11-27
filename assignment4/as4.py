import numpy as np
import hiive.mdptoolbox.example
import gym
import matplotlib.pyplot as plt
import time
import pandas as pd
import re
import copy


"""Markov Decision Process (MDP) Toolbox: ``mdp`` module
=====================================================
The ``mdp`` module provides classes for the resolution of descrete-time Markov
Decision Processes.
Available classes
-----------------
:class:`~mdptoolbox.mdp.MDP`
    Base Markov decision process class
:class:`~mdptoolbox.mdp.FiniteHorizon`
    Backwards induction finite horizon MDP
:class:`~mdptoolbox.mdp.PolicyIteration`
    Policy iteration MDP
:class:`~mdptoolbox.mdp.PolicyIterationModified`
    Modified policy iteration MDP
:class:`~mdptoolbox.mdp.QLearning`
    Q-learning MDP
:class:`~mdptoolbox.mdp.RelativeValueIteration`
    Relative value iteration MDP
:class:`~mdptoolbox.mdp.ValueIteration`
    Value iteration MDP
:class:`~mdptoolbox.mdp.ValueIterationGS`
    Gauss-Seidel value iteration MDP
"""

# Copyright (c) 2011-2015 Steven A. W. Cordwell
# Copyright (c) 2009 INRA
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the <ORGANIZATION> nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import hiive.mdptoolbox.util as _util

_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
                     "condition."
_MSG_STOP_EPSILON_OPTIMAL_POLICY = "Iterating stopped, epsilon-optimal " \
                                   "policy found."
_MSG_STOP_EPSILON_OPTIMAL_VALUE = "Iterating stopped, epsilon-optimal value " \
                                  "function found."
_MSG_STOP_UNCHANGING_POLICY = "Iterating stopped, unchanging policy found."


def _computeDimensions(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))


class MDP:
    """A Markov Decision Problem.
    Let ``S`` = the number of states, and ``A`` = the number of acions.
    Parameters
    ----------
    transitions : array
        Transition probability matrices. These can be defined in a variety of
        ways. The simplest is a numpy array that has the shape ``(A, S, S)``,
        though there are other possibilities. It can be a tuple or list or
        numpy object array of length ``A``, where each element contains a numpy
        array or matrix that has the shape ``(S, S)``. This "list of matrices"
        form is useful when the transition matrices are sparse as
        ``scipy.sparse.csr_matrix`` matrices can be used. In summary, each
        action's transition matrix must be indexable like ``transitions[a]``
        where ``a`` ∈ {0, 1...A-1}, and ``transitions[a]`` returns an ``S`` ×
        ``S`` array-like object.
    reward : array
        Reward matrices or vectors. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
        of lists can be used, where each inner list has length ``S`` and the
        outer list has length ``A``. A list of numpy arrays is possible where
        each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
        or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
        numpy arrays. In addition, the outer list can be replaced by any object
        that can be indexed like ``reward[a]`` such as a tuple or numpy object
        array of length ``A``.
    gamma : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against ``epsilon``. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
        the case where the algorithm does not use an epsilon-optimal stopping
        criterion.
    max_iter : int
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. This must be greater than 0 if
        specified. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a maximum number of iterations.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.
    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    gamma : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.
    Methods
    -------
    run
        Implemented in child classes as the main algorithm loop. Raises an
        exception if it has not been overridden.
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on
    """

    def __init__(self, transitions, reward, gamma, epsilon, max_iter,
                 skip_check=False):
        # Initialise a MDP based on the input parameters.

        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if gamma is not None:
            self.gamma = float(gamma)
            assert 0.0 < self.gamma <= 1.0, (
                "Discount rate must be in ]0; 1]"
            )
            if self.gamma == 1:
                print("WARNING: check conditions of convergence. With no "
                      "discount, convergence can not be assumed.")

        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "The maximum number of iterations must be greater than 0."
            )

        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        if not skip_check:
            # We run a check on P and R to make sure they are describing an
            # MDP. If an exception isn't raised then they are assumed to be
            # correct.
            _util.check(transitions, reward)

        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)
        self.R = self._computeReward(reward, transitions)

        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return (P_repr + "\n" + R_repr)

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                                                            "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.gamma * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def _computeTransition(self, transition):
        return tuple(transition[a] for a in range(self.A))

    def _computeReward(self, reward, transition):
        # Compute the reward for the system in one state chosing an action.
        # Arguments
        # Let S = number of states, A = number of actions
        # P could be an array with 3 dimensions or  a cell array (1xA),
        # each cell containing a matrix (SxS) possibly sparse
        # R could be an array with 3 dimensions (SxSxA) or  a cell array
        # (1xA), each cell containing a sparse matrix (SxS) or a 2D
        # array(SxA) possibly sparse
        try:
            if reward.ndim == 1:
                return self._computeVectorReward(reward)
            elif reward.ndim == 2:
                return self._computeArrayReward(reward)
            else:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
        except (AttributeError, ValueError):
            if len(reward) == self.A:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
            else:
                return self._computeVectorReward(reward)

    def _computeVectorReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            r = _np.array(reward).reshape(self.S)
            return tuple(r for a in range(self.A))

    def _computeArrayReward(self, reward):
        if _sp.issparse(reward):
            raise NotImplementedError
        else:
            def func(x):
                return _np.array(x).reshape(self.S)

            return tuple(func(reward[:, a]) for a in range(self.A))

    def _computeMatrixReward(self, reward, transition):
        if _sp.issparse(reward):
            # An approach like this might be more memory efficeint
            # reward.data = reward.data * transition[reward.nonzero()]
            # return reward.sum(1).A.reshape(self.S)
            # but doesn't work as it is.
            return reward.multiply(transition).sum(1).A.reshape(self.S)
        elif _sp.issparse(transition):
            return transition.multiply(reward).sum(1).A.reshape(self.S)
        else:
            return _np.multiply(transition, reward).sum(1).reshape(self.S)

    def _startRun(self):
        if self.verbose:
            _printVerbosity('Iteration', 'Variation')

        self.time = _time.time()

    def _endRun(self):
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time

    def run(self):
        """Raises error because child classes should implement this function.
        """
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True


class PolicyIteration(MDP):
    """A discounted MDP solved using the policy iteration algorithm.
    Arguments
    ---------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    gamma : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    policy0 : array, optional
        Starting policy.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        class for details. Default is 1000.
    eval_type : int or string, optional
        Type of function used to evaluate policy. 0 or "matrix" to solve as a
        set of linear equations. 1 or "iterative" to solve iteratively.
        Default: 0.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.
    Data Attributes
    ---------------
    V : tuple
        value function
    policy : tuple
        optimal policy
    iter : int
        number of done iterations
    time : float
        used CPU time
    Notes
    -----
    In verbose mode, at each iteration, displays the number
    of differents actions between policy n-1 and n
    Examples
    --------
    >>> import hiive.mdptoolbox, hiive.mdptoolbox.example
    >>> P, R = mdptoolbox.example.rand(10, 3)
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()
    >>> P, R = mdptoolbox.example.forest()
    >>> pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    >>> pi.run()
    >>> expected = (26.244000000000014, 29.484000000000016, 33.484000000000016)
    >>> all(expected[k] - pi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> pi.policy
    (0, 0, 0)
    """

    def __init__(self, transitions, reward, gamma, policy0=None,
                 max_iter=1000, eval_type=0, skip_check=False,
                 run_stat_frequency=None):
        # Initialise a policy iteration MDP.
        #
        # Set up the MDP, but don't need to worry about epsilon values
        MDP.__init__(self, transitions, reward, gamma, None, max_iter,
                     skip_check=skip_check)
        # Check if the user has supplied an initial policy. If not make one.
        self.run_stats = None
        if policy0 is None:
            # Initialise the policy to the one which maximises the expected
            # immediate reward
            null = _np.zeros(self.S)
            self.policy, null = self._bellmanOperator(null)
            del null
        else:
            # Use the policy that the user supplied
            # Make sure it is a numpy array
            policy0 = _np.array(policy0)
            # Make sure the policy is the right size and shape
            assert policy0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                "'policy0' must a vector with length S."
            # reshape the policy to be a vector
            policy0 = policy0.reshape(self.S)
            # The policy can only contain integers between 0 and S-1
            msg = "'policy0' must be a vector of integers between 0 and S-1."
            assert not _np.mod(policy0, 1).any(), msg
            assert (policy0 >= 0).all(), msg
            assert (policy0 < self.S).all(), msg
            self.policy = policy0
        # set the initial values to zero
        self.V = _np.zeros(self.S)
        self.error_mean = []
        self.v_mean = []
        self.p_cumulative = []
        self.run_stat_frequency = max(1, max_iter // 10000) if run_stat_frequency is None else run_stat_frequency

        # Do some setup depending on the evaluation type
        if eval_type in (0, "matrix"):
            self.eval_type = "matrix"
        elif eval_type in (1, "iterative"):
            self.eval_type = "iterative"
        else:
            raise ValueError("'eval_type' should be '0' for matrix evaluation "
                             "or '1' for iterative evaluation. The strings "
                             "'matrix' and 'iterative' can also be used.")

    def _computePpolicyPRpolicy(self):
        # Compute the transition matrix and the reward matrix for a policy.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix
        #     P could be an array with 3 dimensions or a cell array (1xA),
        #     each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #     R could be an array with 3 dimensions (SxSxA) or
        #     a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #     a 2D array(SxA) possibly sparse
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Ppolicy(SxS)  = transition matrix for policy
        # PRpolicy(S)   = reward matrix for policy
        #
        Ppolicy = _np.empty((self.S, self.S))
        Rpolicy = _np.zeros(self.S)
        for aa in range(self.A):  # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                # PR = self._computePR() # an apparently uneeded line, and
                # perhaps harmful in this implementation c.f.
                # mdp_computePpolicyPRpolicy.m
                Rpolicy[ind] = self.R[aa][ind]
        # self.R cannot be sparse with the code in its current condition, but
        # it should be possible in the future. Also, if R is so big that its
        # a good idea to use a sparse matrix for it, then converting PRpolicy
        # from a dense to sparse matrix doesn't seem very memory efficient
        if type(self.R) is _sp.csr_matrix:
            Rpolicy = _sp.csr_matrix(Rpolicy)
        # self.Ppolicy = Ppolicy
        # self.Rpolicy = Rpolicy
        return Ppolicy, Rpolicy

    def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        # Evaluate a policy using iteration.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA)  = transition matrix
        #    P could be an array with 3 dimensions or
        #    a cell array (1xS), each cell containing a matrix possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #    R could be an array with 3 dimensions (SxSxA) or
        #    a cell array (1xA), each cell containing a sparse matrix (SxS) or
        #    a 2D array(SxA) possibly sparse
        # discount  = discount rate in ]0; 1[
        # policy(S) = a policy
        # V0(S)     = starting value function, optional (default : zeros(S,1))
        # epsilon   = epsilon-optimal policy search, upper than 0,
        #    optional (default : 0.0001)
        # max_iter  = maximum number of iteration to be done, upper than 0,
        #    optional (default : 10000)
        #
        # Evaluation
        # ----------
        # Vpolicy(S) = value function, associated to a specific policy
        #
        # Notes
        # -----
        # In verbose mode, at each iteration, displays the condition which
        # stopped iterations: epsilon-optimum value function found or maximum
        # number of iterations reached.
        #
        try:
            assert V0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                "'V0' must be a vector of length S."
            policy_V = _np.array(V0).reshape(self.S)
        except AttributeError:
            if V0 == 0:
                policy_V = _np.zeros(self.S)
            else:
                policy_V = _np.array(V0).reshape(self.S)

        policy_P, policy_R = self._computePpolicyPRpolicy()

        if self.verbose:
            _printVerbosity("Iteration", "V variation")

        itr = 0
        done = False

        while not done:
            itr += 1

            Vprev = policy_V
            policy_V = policy_R + self.gamma * policy_P.dot(Vprev)

            variation = _np.absolute(policy_V - Vprev).max()
            if self.verbose:
                _printVerbosity(itr, variation)

            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.gamma) / self.gamma) * epsilon:
                done = True
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_VALUE)
            elif itr == max_iter:
                done = True
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)

        self.V = policy_V
        return policy_V, policy_R, itr

    def _evalPolicyMatrix(self):
        # Evaluate the value function of the policy using linear equations.
        #
        # Arguments
        # ---------
        # Let S = number of states, A = number of actions
        # P(SxSxA) = transition matrix
        #      P could be an array with 3 dimensions or a cell array (1xA),
        #      each cell containing a matrix (SxS) possibly sparse
        # R(SxSxA) or (SxA) = reward matrix
        #      R could be an array with 3 dimensions (SxSxA) or
        #      a cell array (1xA), each cell containing a sparse matrix (SxS)
        #      or a 2D array(SxA) possibly sparse
        # discount = discount rate in ]0; 1[
        # policy(S) = a policy
        #
        # Evaluation
        # ----------
        # Vpolicy(S) = value function of the policy
        #
        Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        policy_V = _np.linalg.solve((_sp.eye(self.S, self.S) - self.gamma * Ppolicy), Rpolicy)
        self.V = policy_V
        return policy_V, Rpolicy, None

    def _build_run_stat(self, i, s, a, r, p, v, error):
        run_stat = {
            'State': s,
            'Action': a,
            'Reward': r,
            'Error': error,
            'Time': _time.time() - self.time,
            'V[0]': v[0],
            'Max V': _np.max(v),
            'Mean V': _np.mean(v),
            'Iteration': i,
            # 'Value': v.copy(),
            # 'Policy': p.copy()
        }
        return run_stat

    def run(self):
        # Run the policy iteration algorithm.
        self._startRun()
        self.run_stats = []

        self.error_mean = []
        error_cumulative = []

        self.v_mean = []
        v_cumulative = []

        self.p_cumulative = []
        run_stats = []
        while True:
            self.iter += 1
            take_run_stat = self.iter % self.run_stat_frequency == 0 or self.iter == self.max_iter
            # these _evalPolicy* functions will update the classes value
            # attribute
            policy_V, policy_R, itr = (self._evalPolicyMatrix()
                                       if self.eval_type == 'matrix'
                                       else self._evalPolicyIterative())

            if take_run_stat:
                v_cumulative.append(policy_V)
                if len(v_cumulative) == 100:
                    self.v_mean.append(_np.mean(v_cumulative, axis=1))
                    v_cumulative = []
                if len(self.p_cumulative) == 0 or not _np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((self.iter, self.policy.copy()))

            # This should update the classes policy attribute but leave the
            # value alone
            policy_next, next_v = self._bellmanOperator()
            error = _np.absolute(next_v - policy_V).max()
            run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(policy_V),
                                                  p=policy_next, v=policy_V, error=error))

            if take_run_stat:
                error_cumulative.append(error)
                if len(error_cumulative) == 100:
                    self.error_mean.append(_np.mean(error_cumulative))
                    error_cumulative = []
                self.run_stats.append(run_stats[-1])
                run_stats = []
            del next_v
            # calculate in how many places does the old policy disagree with
            # the new policy
            nd = (policy_next != self.policy).sum()
            # if verbose then continue printing a table
            if self.verbose:
                _printVerbosity(self.iter, nd)
            # Once the policy is unchanging of the maximum number of
            # of iterations has been reached then stop

            # Error, rewards, and time for every iteration and number of PI steps which might be specific to my setup
            if nd == 0:
                if self.verbose:
                    print(_MSG_STOP_UNCHANGING_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break
            else:
                self.policy = policy_next

        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(_np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(_np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats



class QLearning(MDP):
    """A discounted MDP solved using the Q learning algorithm.
    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    gamma : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an
        integer greater than the default value. Defaut: 10,000.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.
    Data Attributes
    ---------------
    Q : array
        learned Q matrix (SxA)
    V : tuple
        learned value function (S).
    policy : tuple
        learned optimal policy (S).
    mean_discrepancy : array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).
    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import hiive.mdptoolbox, hiive.mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    >>> ql.run()
    >>> ql.Q
    array([[ 11.198909  ,  10.34652034],
           [ 10.74229967,  11.74105792],
           [  2.86980001,  12.25973286]])
    >>> expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (0, 1, 1)
    >>> import hiive.mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> np.random.seed(0)
    >>> ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
    >>> ql.run()
    >>> ql.Q
    array([[ 33.33010866,  40.82109565],
           [ 34.37431041,  29.67236845]])
    >>> expected = (40.82109564847122, 34.37431040682546)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (1, 0)
    """

    def __init__(self, transitions, reward, gamma,
                 alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                 n_iter=10000, skip_check=False, iter_callback=None,
                 run_stat_frequency=None):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            # We don't want to send this to MDP because _computePR should not
            #  be run on it, so check that it defines an MDP
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.alpha = _np.clip(alpha, 0., 1.)
        self.alpha_start = self.alpha
        self.alpha_decay = _np.clip(alpha_decay, 0., 1.)
        self.alpha_min = _np.clip(alpha_min, 0., 1.)
        self.gamma = _np.clip(gamma, 0., 1.)
        self.epsilon = _np.clip(epsilon, 0., 1.)
        self.epsilon_start = self.epsilon
        self.epsilon_decay = _np.clip(epsilon_decay, 0., 1.)
        self.epsilon_min = _np.clip(epsilon_min, 0., 1.)

        # Initialisations
        self.Q = _np.zeros((self.S, self.A))

        self.run_stats = []
        self.error_mean = []
        self.v_mean = []
        self.p_cumulative = []
        self.iter_callback = iter_callback
        self.S_freq = _np.zeros((self.S, self.A))
        self.run_stat_frequency = max(1, self.max_iter // 10000) if run_stat_frequency is None else run_stat_frequency

    def run(self):

        # Run the Q-learning algorithm.
        error_cumulative = []
        self.run_stats = []
        self.error_mean = []

        v_cumulative = []
        self.v_mean = []

        self.p_cumulative = []

        self.time = _time.time()

        # initial state choice
        s = _np.random.randint(0, self.S)
        reset_s = False
        run_stats = []
        for n in range(1, self.max_iter + 1):

            take_run_stat = n % self.run_stat_frequency == 0 or n == self.max_iter

            # Reinitialisation of trajectories every 100 transitions
            if (self.iter_callback is None and (n % 100) == 0) or reset_s:
                s = _np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            # The agent takes random actions for probability ε and greedy action for probability (1-ε).
            pn = _np.random.random()
            if pn < self.epsilon:
                a = _np.random.randint(0, self.A)
            else:
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Q[s, a] = Q[s, a] + alpha*(R + gamma*Max[Q(s’, A)] - Q[s, a])
            # Updating the value of Q
            dQ = self.alpha * (r + self.gamma * self.Q[s_new, :].max() - self.Q[s, a])
            self.Q[s, a] = self.Q[s, a] + dQ

            # Computing means all over maximal Q variations values
            error = _np.absolute(dQ)

            # compute the value function and the policy
            v = self.Q.max(axis=1)
            self.V = v
            p = self.Q.argmax(axis=1)
            self.policy = p

            self.S_freq[s,a] += 1
            run_stats.append(self._build_run_stat(i=n, s=s, a=a, r=_np.max(self.V), p=p, v=v, error=error))

            if take_run_stat:
                error_cumulative.append(error)

                if len(error_cumulative) == 100:
                    self.error_mean.append(_np.mean(error_cumulative))
                    error_cumulative = []

                v_cumulative.append(v)

                if len(v_cumulative) == 100:
                    self.v_mean.append(_np.mean(v_cumulative, axis=1))
                    v_cumulative = []

                if len(self.p_cumulative) == 0 or not _np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((n, self.policy.copy()))
                """
                Rewards,errors time at each iteration I think
                But that’s for all of them and steps per episode?
                Alpha decay and min ?
                And alpha and epsilon at each iteration?
                """
                self.run_stats.append(run_stats[-1])
                run_stats = []

            if self.iter_callback is not None:
                reset_s = self.iter_callback(s, a, s_new)

            # current state is updated
            s = s_new

            self.alpha *= self.alpha_decay
            if self.alpha < self.alpha_min:
                self.alpha = self.alpha_min

            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        self._endRun()
        # add stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(_np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(_np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats

    def _build_run_stat(self, i, a, error, p, r, s, v):
        run_stat = {
            'State': s,
            'Action': a,
            'Reward': r,
            'Error': error,
            'Time': _time.time() - self.time,
            'Alpha': self.alpha,
            'Epsilon': self.epsilon,
            'Gamma': self.gamma,
            'V[0]': v[0],
            'Max V': _np.max(v),
            'Mean V': _np.mean(v),
            'Iteration': i,
            # 'Value': v.copy(),
            # 'Policy': p.copy()
        }
        return run_stat



class ValueIteration(MDP):
    """A discounted MDP solved using the value iteration algorithm.
    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations.
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.
    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    gamma : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.
    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.
    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.
    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.
    Examples
    --------
    >>> import hiive.mdptoolbox, hiive.mdptoolbox.example
    >>> P, R = hiive.mdptoolbox.example.forest()
    >>> vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> expected = (5.93215488, 9.38815488, 13.38815488)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4
    >>> import hiive.mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26
    >>> import hiive.mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    """

    def __init__(self, transitions, reward, gamma, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False,
                 run_stat_frequency=None):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions, reward, gamma, epsilon, max_iter,
                     skip_check=skip_check)
        self.run_stats = None
        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                                                 "a vector of length S."
            self.V = _np.array(initial_value).reshape(self.S)
        if self.gamma < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon
        self.v_mean = []
        self.error_mean = []
        self.p_cumulative = []
        self.run_stat_frequency = max(1, self.max_iter // 10000) if run_stat_frequency is None else run_stat_frequency

    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        k = 0
        h = _np.zeros(self.S)

        for ss in range(self.S):
            PP = _np.zeros((self.A, self.S))
            for aa in range(self.A):
                try:
                    PP[aa] = self.P[aa][:, ss]
                except ValueError:
                    PP[aa] = self.P[aa][:, ss].todense().A1
            # minimum of the entire array.
            h[ss] = PP.min()

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        span = _util.getSpan(value - Vprev)
        max_iter = (_math.log((epsilon * (1 - self.gamma) / self.gamma) /
                              span) / _math.log(self.gamma * k))
        # self.V = Vprev

        self.max_iter = int(_math.ceil(max_iter))

    def run(self):
        # Run the value iteration algorithm.
        self._startRun()
        self.run_stats = []
        error_cumulative = []
        v_cumulative = []
        self.v_mean = []
        self.error_mean = []
        self.p_cumulative = []
        run_stats = []
        while True:
            self.iter += 1
            take_run_stat = self.iter % self.run_stat_frequency == 0 or self.iter == self.max_iter

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            error = _util.getSpan(self.V - Vprev)
            run_stats.append(self._build_run_stat(i=self.iter, s=None, a=None, r=_np.max(self.V),
                                                           p=self.policy, v=self.V, error=error))
            if take_run_stat:
                error_cumulative.append(error)
                if len(self.p_cumulative) == 0 or not _np.array_equal(self.policy, self.p_cumulative[-1][1]):
                    self.p_cumulative.append((self.iter, self.policy.copy()))
                if len(v_cumulative) == 100:
                    self.v_mean.append(_np.mean(v_cumulative, axis=1))
                    v_cumulative = []
                if len(error_cumulative) == 100:
                    self.error_mean.append(_np.mean(error_cumulative))
                    error_cumulative = []

                self.run_stats.append(run_stats[-1])
                run_stats = []

            if self.verbose:
                _printVerbosity(self.iter, error)

            if error < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self._endRun()

        # catch stragglers
        if len(v_cumulative) > 0:
            self.v_mean.append(_np.mean(v_cumulative, axis=1))
        if len(error_cumulative) > 0:
            self.error_mean.append(_np.mean(error_cumulative))
        if self.run_stats is None or len(self.run_stats) == 0:
            self.run_stats = run_stats
        return self.run_stats

    def _build_run_stat(self, i, s, a, r, p, v, error):
        run_stat = {
            'State': None,
            'Action': None,
            'Reward': r,
            'Error': error,
            'Time': _time.time() - self.time,
            # 'Epsilon': self.epsilon,
            'Max V': _np.max(v),
            'Mean V': _np.mean(v),
            'Iteration': i,

            # 'Value': v.copy(),
            # 'Policy': p.copy()
        }
        return run_stat


class OpenAI_MDPToolbox:

    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments. 
    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control
    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.  
    """
    
    def __init__(self, openAI_env_name:str, render:bool=False, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class
        :param openAI_env_name: Valid name of an Open AI Gym env 
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean 
        """
        self.env_name = openAI_env_name
    
        self.env = gym.make(self.env_name, **kwargs)
        self.env.reset()

        if render:
            self.env.render()
        
        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR()
        
    def convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob*self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob

# DFS to check that it's a valid path.
def is_valid(board: [], max_size: int):
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

def generate_random_map(size: int = 8, p: float = 0.8):
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


def makePlot(x, y, xLbl, yLbl, title, size=(4,3)):
    plt.rcParams["figure.figsize"] = size
    plt.xlabel(xLbl)
    plt.ylabel(yLbl)
    plt.title(title)
    plt.plot(x, y)
    plt.show()
    plt.close()
    
def make_Graph_Dict(run_stats, var):
    times = []
    graphDict = {v:[] for v in var}
    graphDict["times"] = times
    for iteration in run_stats:
        times.append(iteration["Time"])
        for v in iteration:
            if v in var:
                graphDict[v].append(iteration[v])
    return graphDict

np.random.seed(0)

flMap = generate_random_map(15)

#################################
#####Small Forest Management#####
#################################
P, R = hiive.mdptoolbox.example.forest(S=10, p=0.3)

#################################
#########Value Iteration#########
#################################
np.random.seed(2)
gammaVal = [.50,.80,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    v = ValueIteration(P, R, g, skip_check=True)
    v.run()
    end = time.time()
    clockTimeVI = end-start
    v.policy
    v.run_stats[-20:]
    vStats = make_Graph_Dict(v.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(vStats)
    
    print(str(g) + " Policy:", v.policy)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Forest Management Value Iteration Max Utility by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()
    

plt.plot(vStats["Iteration"],vStats["Mean V"], label='Mean V')
plt.plot(vStats["Iteration"],vStats["Max V"], label='Max V')
plt.title('Forest Management Value Iteration Max V by Iter SMALL SPACE')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

#################################
########Policy Iteration#########
#################################
np.random.seed(6)
gammaVal = [.50,.80,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    if g == .99:
        p = PolicyIteration(P, R, .99, eval_type=0)
    elif g == .95:
        p = PolicyIteration(P, R, .93, eval_type=0)
    else:
        p = PolicyIteration(P, R, g, eval_type=0)
    p.run()
    end = time.time()
    clockTimePI = end-start
    p.policy
    p.run_stats[-20:]
    pStats = p.run_stats
    pStats = make_Graph_Dict(p.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(pStats)
    print(str(g) + " Policy:", p.policy)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Forest Management Policy Iteration Max Utility by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(pStats["Iteration"],pStats["Mean V"], label='Mean V')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Max V')
plt.title('Forest Management Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()



plt.plot(vStats["Iteration"],vStats["Max V"], label='Value Iteration')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Policy Iteration')
plt.title('Forest Management Long Term Reward by Iter')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(vStats["Iteration"],vStats["Time"], label='Value Clock Time')
plt.plot(pStats["Iteration"],pStats["Time"], label='Policy Clock Time')
plt.title('Forest Management Clock Time by Iter (Policy Linear Solver)')
plt.xlabel('Iteration')
plt.ylabel('Clock Time')
plt.legend()
plt.show()
plt.close()


#################################
###########Q Learning###########
#################################
P, R = hiive.mdptoolbox.example.forest(S=205, p=0.03)

for i in range(0,205):
    percentile = i/205
    
    if percentile <= .34:
        R[i,1] = 0.0
    elif percentile <= .67:
        R[i,1] = 1.0
    elif percentile <=.9:
        R[i,1] = 2.0
    elif i < 195:
        R[i,1] = 3.0
    else:
        R[i,1] = 5.0
        
np.random.seed(2)
decayRate = [.90,.95,.999]
stats = []
for d in decayRate:
    start = time.time()
    q = QLearning(P, R, 0.95, epsilon=1,epsilon_decay=d, n_iter=10000000, skip_check=True)
    q.run()
    end = time.time()
    clockTimeQI = end-start
    print(d, q.policy)
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(qStats)


count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Decay'))
    count += 1

plt.title('LARGE Forest Mgmt Q Learning Rate by Decay')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.legend()
plt.show()
plt.close()


np.random.seed(2)
decayRate = [.80,.95,.99]
stats = []
for d in decayRate:
    start = time.time()
    if d == .99:
        q = QLearning(P, R, .98, epsilon=1,epsilon_decay=.99, n_iter=10000000, skip_check=True)
    else:
        q = QLearning(P, R, d, epsilon=1,epsilon_decay=.99, n_iter=10000000, skip_check=True)
    
    q.run()
    end = time.time()
    clockTimeQI = end-start
    print(d, q.policy)
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(qStats)


count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Gamma'))
    count += 1

plt.title('SMALL Forest Mgmt Q Learning Rate by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.legend()
plt.show()
plt.close()


qStatsS = copy.copy(qStats)
plt.plot(qStats["Iteration"],qStatsS["Time"], label='Small State')
plt.plot(qStats["Iteration"],qStats["Time"], label='Large State')
plt.title('Forest Mgmt Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Clock Time in Seconds')
plt.legend()
plt.show()
plt.close()


#Large Forest Management
P, R = hiive.mdptoolbox.example.forest(S=205, p=0.03)

for i in range(0,205):
    percentile = i/205
    
    if percentile <= .34:
        R[i,1] = 0.0
    elif percentile <= .67:
        R[i,1] = 1.0
    elif percentile <=.9:
        R[i,1] = 2.0
    elif i < 195:
        R[i,1] = 3.0
    else:
        R[i,1] = 5.0
    

#################################
#########Value Iteration#########
#################################
np.random.seed(2)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    v = ValueIteration(P, R, g, skip_check=True)
    v.run()
    end = time.time()
    clockTimeVI = end-start
    v.policy
    v.run_stats[-20:]
    vStats = make_Graph_Dict(v.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(vStats)
    
    print(str(g) + " Policy:","\n", v.policy)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('LARGE Forest Mgmt Value Iteration Reward by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()
    

num_iters = len(vStats["Mean V"])
#makePlot(vStats["Iteration"], vStats["Max V"], "Iteration", "Max Value", "Forest Management Value Iteration Max V by Iter.", size=(5,3))

plt.plot(vStats["Iteration"],vStats["Mean V"], label='Mean V')
plt.plot(vStats["Iteration"],vStats["Max V"], label='Max V')
plt.title('Forest Management Value Iteration Max V by Iter LARGE SPACE')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

#################################
########Policy Iteration#########
#################################
np.random.seed(3)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    p = PolicyIteration(P, R, g, eval_type=0)
    p.run()
    end = time.time()
    clockTimePI = end-start
    p.policy
    p.run_stats[-20:]
    pStats = p.run_stats
    pStats = make_Graph_Dict(p.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(pStats)
    print(str(g) + " Policy:", "\n", p.policy)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('LARGE Forest Mgmt Value Iteration Reward by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(pStats["Iteration"],pStats["Mean V"], label='Mean V')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Max V')
plt.title('LARGE Forest Mgmt Policy Iteration Reward by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()



plt.plot(vStats["Iteration"],vStats["Max V"], label='Value Clock Time')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Policy Clock Time')
plt.title('Forest Management Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()

plt.plot(vStats["Iteration"],vStats["Time"], label='Value Clock Time')
plt.plot(pStats["Iteration"],pStats["Time"], label='Policy Clock Time')
plt.title('Forest Management Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()


#################################
###########Q Learning###########
#################################
np.random.seed(2)
decayRate = [.90,.95,.99]
stats = []
for d in decayRate:
    start = time.time()
    q = QLearning(P, R, 0.95, epsilon=1,epsilon_decay=d, n_iter=1000000, skip_check=True)
    q.run()
    end = time.time()
    clockTimeQI = end-start
    print(d, q.policy)
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])
    stats.append(qStats)

plt.plot(qStats["Iteration"],qStats["Max V"], label='Mean V')
plt.plot(qStats["Iteration"],qStats["Reward"], label='Reward')
plt.title('Forest Management Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Decay'))
    count += 1

plt.title('Forest Management Q Learning Rate by Decay')
plt.xlabel('Iteration')
plt.ylabel('Mean V Value')
plt.legend()
plt.show()
plt.close()









#################################
########Frozen Lake##############
#################################
np.random.seed(0)
flMap = generate_random_map(15)

np.random.seed(0)
flMap = generate_random_map(3)
flMap

outMap = ''
for l in flMap:
    for c in l:
        outMap += c + ","
    outMap += ("\n")
print(outMap)
        

#################################
##Small Frozen Lake Management###
#################################
np.random.seed(0)
flMap = generate_random_map(4)
fl = OpenAI_MDPToolbox('FrozenLake-v1', desc=flMap, is_slippery=True)
#fl = OpenAI_MDPToolbox('FrozenLake-v1', desc=flMap, is_slippery=True)

P, R = fl.P, fl.R

#################################
#########Value Iteration#########
#################################
np.random.seed(2)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    v = ValueIteration(P, R, g, skip_check=True)
    v.run()
    end = time.time()
    clockTimeVI = end-start
    v.policy
    v.run_stats[-20:]
    vStats = make_Graph_Dict(v.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(vStats)
    
    print(str(g) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = v.policy[counter]
            
            if direct == 0:
                arrow = '<'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '>'
            else:
                arrow = '^'
            
            outMap += c + '  ' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
            
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Frozen Lake Value Iteration Reward by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()
    

plt.plot(vStats["Iteration"],vStats["Mean V"], label='Mean V')
plt.plot(vStats["Iteration"],vStats["Max V"], label='Max V')
plt.title('Forest Management Value Iteration Max V by Iter SMALL SPACE')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

#################################
########Policy Iteration#########
#################################
np.random.seed(3)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    p = PolicyIteration(P, R, g, eval_type=0)
    p.run()
    end = time.time()
    clockTimePI = end-start
    p.policy
    p.run_stats[-20:]
    pStats = p.run_stats
    pStats = make_Graph_Dict(p.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(pStats)
    
    print(str(g) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = p.policy[counter]
            
            if direct == 0:
                arrow = '<'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '>'
            else:
                arrow = '^'
            
            outMap += c + '  ' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Frozen Lake Policy Iteration Max Utility by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(pStats["Iteration"],pStats["Mean V"], label='Mean V')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Max V')
plt.title('Forest Management Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()



plt.plot(vStats["Iteration"],vStats["Max V"], label='Value Iteration')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Policy Iteration')
plt.title('Frozen Lake Policy Iteration Max Reward by Iter')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(vStats["Iteration"],vStats["Time"], label='Value Iteration')
plt.plot(pStats["Iteration"],pStats["Time"], label='Policy Iteration')
plt.title('Frozen Lake Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Clock Time')
plt.legend()
plt.show()
plt.close()


#################################
###########Q Learning###########
#################################
np.random.seed(2)
decayRate = [.90,.95,.99,.999]
stats = []
for d in decayRate:
    start = time.time()
    q = QLearning(P, R, 0.99, epsilon=1,epsilon_decay=d, n_iter=10000000, skip_check=True)
    q.run()
    end = time.time()
    clockTimeQI = end-start
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])
    stats.append(qStats)
    
    print(str(d) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = q.policy[counter]
            
            if direct == 0:
                arrow = '<'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '>'
            else:
                arrow = '^'
            
            outMap += c + '  ' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
    
    
start = time.time()
q = QLearning(P, R, 0.99, epsilon=1,epsilon_decay=.999, n_iter=10000000, skip_check=True)
q.run()
end = time.time()
clockTimeQI = end-start
qStats = q.run_stats
qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])

print(str(d) + " Policy:")
outMap = ''
counter = 0
for l in flMap:
    for c in l:
        direct = q.policy[counter]
        
        if direct == 0:
            arrow = '<-'
        elif direct == 1:
            arrow = 'v'
        elif direct == 2:
            arrow = '->'
        else:
            arrow = '^'
        
        outMap += c + ':' + arrow + ','
        counter += 1
        
    outMap += '\n'
print(outMap)

plt.plot(qStats["Iteration"],qStats["Max V"], label='Max V')
plt.plot(qStats["Iteration"],qStats["Mean V"], label='Mean V')
plt.title('Forest Management Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Decay'))
    count += 1

plt.title('Forest Management Q Learning Rate by Decay')
plt.xlabel('Iteration')
plt.ylabel('Mean V Value')
plt.legend()
plt.show()
plt.close()




#################################
###########Q Learning TWOOOOOOOO###########
#################################
np.random.seed(0)
flMap = generate_random_map(4)
#flMap = generate_random_map(15)
fl = OpenAI_MDPToolbox('FrozenLake-v1', desc=flMap, is_slippery=True)

P, R = fl.P, fl.R
        
np.random.seed(2)
decayRate = [.9,.95,.99]
stats = []
for d in decayRate:
    start = time.time()
    q = QLearning(P, R, 0.95, epsilon=1,epsilon_decay=d, n_iter=1000000, skip_check=True)
    q.run()
    end = time.time()
    clockTimeQI = end-start
    print(d, q.policy)
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(qStats)

count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Decay'))
    count += 1

plt.title('SMALL Frozen Lake Q Learning Rate by Decay')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.legend()
plt.show()
plt.close()


np.random.seed(2)
decayRate = [.80,.95,.99]
stats = []
for d in decayRate:
    start = time.time()
    if d == .99:
        q = QLearning(P, R, .98, epsilon=1,epsilon_decay=.99, n_iter=100000, skip_check=True)
    else:
        q = QLearning(P, R, d, epsilon=1,epsilon_decay=.99, n_iter=100000, skip_check=True)
    
    q.run()
    end = time.time()
    clockTimeQI = end-start
    print(d, q.policy)
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(qStats)


count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Gamma'))
    count += 1

plt.title('SMALL Frozen Lake Q Learning Rate by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.legend()
plt.show()
plt.close()


qStatsS = copy.copy(qStats)
plt.plot(qStats["Iteration"],qStatsS["Time"], label='Small State')
plt.plot(qStats["Iteration"],qStats["Time"], label='Large State')
plt.title('Forest Mgmt Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Clock Time in Seconds')
plt.legend()
plt.show()
plt.close()




start = time.time()
q = QLearning(P, R, 0.8, epsilon=1,epsilon_decay=.95, n_iter=100000000, skip_check=True)
q.run()
end = time.time()
clockTimeQI = end-start
qStats = q.run_stats
qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])

print(str(d) + " Policy:")
outMap = ''
counter = 0
for l in flMap:
    for c in l:
        direct = q.policy[counter]
        
        if direct == 0:
            arrow = '<'
        elif direct == 1:
            arrow = 'v'
        elif direct == 2:
            arrow = '>'
        else:
            arrow = '^'
        
        outMap += c + ' ' + arrow + ','
        counter += 1
        
    outMap += '\n'
print(outMap)

#################################
##Large Frozen Lake Management###
#################################
np.random.seed(0)
flMap = generate_random_map(15)
fl = OpenAI_MDPToolbox('FrozenLake-v1', desc=flMap, is_slippery=True)
#fl = OpenAI_MDPToolbox('FrozenLake-v1', desc=flMap, is_slippery=True)

P, R = fl.P, fl.R

#################################
#########Value Iteration#########
#################################
np.random.seed(2)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    v = ValueIteration(P, R, g, skip_check=True)
    v.run()
    end = time.time()
    clockTimeVI = end-start
    v.policy
    v.run_stats[-20:]
    vStats = make_Graph_Dict(v.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(vStats)
    
    print(str(g) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = v.policy[counter]
            
            if direct == 0:
                arrow = '<'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '>'
            else:
                arrow = '^'
            
            outMap += c + ' ' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
            
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Frozen Lake Value Iteration Max Reward by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()
    

plt.plot(vStats["Iteration"],vStats["Mean V"], label='Mean V')
plt.plot(vStats["Iteration"],vStats["Max V"], label='Max V')
plt.title('Frozen Lake Value Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

#################################
########Policy Iteration#########
#################################
np.random.seed(3)
gammaVal = [.5,.8,.95,.99]
stats = []

for  g in gammaVal:
    start = time.time()
    p = PolicyIteration(P, R, g, eval_type=1)
    p.run()
    end = time.time()
    clockTimePI = end-start
    p.policy
    p.run_stats[-20:]
    pStats = p.run_stats
    pStats = make_Graph_Dict(p.run_stats, ["Mean V", "Max V", "Iteration", "Reward", "Time"])
    stats.append(pStats)
    
    print(str(g) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = p.policy[counter]
            
            if direct == 0:
                arrow = '<'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '>'
            else:
                arrow = '^'
            
            outMap += c + ' ' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
    
count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(gammaVal[count]) + (' Gamma'))
    count += 1

plt.title('Frozen Lake Policy Iteration Max Reward by Gamma')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(pStats["Iteration"],pStats["Mean V"], label='Mean V')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Max V')
plt.title('Frozen Lake Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('Utility Value')
plt.legend()
plt.show()
plt.close()



plt.plot(vStats["Iteration"],vStats["Max V"], label='Value Iteration')
plt.plot(pStats["Iteration"],pStats["Max V"], label='Policy Iteration')
plt.title('Frozen Lake Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('Max Reward')
plt.legend()
plt.show()
plt.close()

plt.plot(vStats["Iteration"],vStats["Time"], label='Value Iteration')
plt.plot(pStats["Iteration"],pStats["Time"], label='Policy Iteration')
plt.title('Frozen Lake Clock Time by Iter')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Clock Time')
plt.legend()
plt.show()
plt.close()


#################################
###########Q Learning###########
#################################
np.random.seed(2)
decayRate = [.90,.95,.99,.999]
stats = []
for d in decayRate:
    start = time.time()
    q = QLearning(P, R, 0.99, epsilon=1,epsilon_decay=d, n_iter=10000000, skip_check=True)
    q.run()
    end = time.time()
    clockTimeQI = end-start
    qStats = q.run_stats
    qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])
    stats.append(qStats)
    
    print(str(d) + " Policy:")
    outMap = ''
    counter = 0
    for l in flMap:
        for c in l:
            direct = q.policy[counter]
            
            if direct == 0:
                arrow = '<-'
            elif direct == 1:
                arrow = 'v'
            elif direct == 2:
                arrow = '->'
            else:
                arrow = '^'
            
            outMap += c + ':' + arrow + ','
            counter += 1
            
        outMap += '\n'
    print(outMap)
    
    
start = time.time()
q = QLearning(P, R, 0.99, epsilon=1,epsilon_decay=.999, n_iter=10000000, skip_check=True)
q.run()
end = time.time()
clockTimeQI = end-start
qStats = q.run_stats
qStats = make_Graph_Dict(q.run_stats, ["Mean V", "Max V", "Iteration", "Reward"])

print(str(d) + " Policy:")
outMap = ''
counter = 0
for l in flMap:
    for c in l:
        direct = q.policy[counter]
        
        if direct == 0:
            arrow = '<-'
        elif direct == 1:
            arrow = 'v'
        elif direct == 2:
            arrow = '->'
        else:
            arrow = '^'
        
        outMap += c + ':' + arrow + ','
        counter += 1
        
    outMap += '\n'
print(outMap)

plt.plot(qStats["Iteration"],qStats["Max V"], label='Max V')
plt.plot(qStats["Iteration"],qStats["Mean V"], label='Mean V')
plt.title('Forest Management Policy Iteration Max V by Iter')
plt.xlabel('Iteration')
plt.ylabel('V Value')
plt.legend()
plt.show()
plt.close()

count = 0
for s in stats:
    plt.plot(s["Iteration"],s["Reward"], label=str(decayRate[count]) + (' Decay'))
    count += 1

plt.title('Forest Management Q Learning Rate by Decay')
plt.xlabel('Iteration')
plt.ylabel('Mean V Value')
plt.legend()
plt.show()
plt.close()


