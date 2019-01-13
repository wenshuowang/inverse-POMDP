'''
This incorporates the oneboxtask_ini and oneboxMDPsolver and oneboxGenerate into one file with oneboxMDP object

'''

from __future__ import division
from boxtask_func import *
#from mdptoolbox import *
from HMMonebox import *
import numpy.matlib
from scipy.linalg import block_diag
from numpy.linalg import inv
from math import floor, ceil
from MDPclass import *

# we need two different transition matrices, one for each of the following actions:
a0 = 0  # a0 = do nothing
pb = 1  # pb  = push button
sigmaTb = 0.1    # variance for the gaussian approximation in belief transition matrix
temperatureQ = 0.061  # temperature for soft policy based on Q value


class oneboxMDP:
    """
    model onebox problem, set up the transition matrices and reward based on the given parameters,
    and solve the MDP problem, return the optimal policy
    """
    def __init__(self, discount, nq, nr, na, parameters):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.n = self.nq * self.nr  # total number of states
        self.parameters = parameters  # [beta, gamma, epsilon, rho]
        self.ThA = np.zeros((self.na, self.n, self.n))
        self.R = np.zeros((self.na, self.n, self.n))

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'oneboxtask_ini.py'
        :return:
                ThA: transition probability,
                     shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                   shape: (# of action) * (# of states, old state) * (# of states, new state)
        """

        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]
        Reward = 1

        # initialize probability distribution over states (belief and world)
        pr0 = np.array([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pb0 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)

        ph0 = kronn(pr0, pb0)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        # Tb = beliefTransitionMatrix(gamma, epsilon, nq, eta)
        # belief transition matrix

        Tb = beliefTransitionMatrixGaussian(gamma, epsilon, self.nq, sigmaTb)
        # softened the belief transition matrix with 2-dimensional Gaussian distribution

        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(Tr, Tb)
        # kronecker product of these transition matrices

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                              np.zeros((self.nq - 2, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                              np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                              np.zeros(((self.nq - 2), 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        self.ThA[pb, :, :] = Trb.dot(self.ThA[a0, :, :])
        #self.ThA[pb, :, :] = Trb

        Reward_h = tensorsumm(np.array([[0, Reward]]), np.zeros((1, self.nq)))
        Reward_a = - np.array([0, pushButtonCost])

        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3
        self.R = Reward

        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T

    def solveMDP_op(self, epsilon = 10**-6, niterations = 10000):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                policy: the optimal policy based on the maximum Q value
                        shape: # of states, take integer values indicating the action
                softpolicy: probability of choosing each action
                            shape: (# of actions) * (# of states)
        """

        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations)
        vi.run()
        self.Q = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.policy = np.array(vi.policy)

        #pi = mdp.ValueIteration(self.ThA, self.R, self.discount, epsilon, niterations)
        #pi.run()
        #self.Q = self._QfromV(pi)
        #self.policy = np.array(pi.policy)


    def solveMDP_sfm(self, epsilon = 10**-6, niterations = 10000, initial_value=0):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                policy: softmax policy
        """

        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        #print self.Qsfm

        return  vi.V


    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].dot(ValueIteration.V)
        return Q




class oneboxMDPdata(oneboxMDP):
    """
    This class generates the data based on the object oneboxMDP. The parameters, and thus the transition matrices and
    the rewrd function, are shared for the oneboxMDP and this data generator class.
    """
    def __init__(self, discount, nq, nr, na, parameters, parametersExp,
                 sampleTime, sampleNum):
        oneboxMDP.__init__(self, discount, nq, nr, na, parameters)

        self.parametersExp = parametersExp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.belief = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState = np.zeros((self.sampleNum, self.sampleTime))

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def dataGenerate_op(self, beliefInitial, rewInitial):
        """
        This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
        as a separate class, since at that time, the oneboxMDP class was not defined.
        In this file, all the functions are implemented under a single class.

        :return: the obseravations
        """

        beta = self.parameters[0]  # available food dropped back into box after button press
        gamma = self.parameters[1]  # reward becomes available
        epsilon = self.parameters[2]  # available food disappears
        rho = self.parameters[3]  # food in mouth is consumed

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]


        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)

                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
                    self.action[i, t] = self.policy[self.hybrid[i, t]]
                            # action is based on optimal policy
                else:
                    if self.action[i, t-1] != pb:
                        stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
                        self.hybrid[i, t] = np.argmax(stattemp)
                        self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
                        self.action[i, t] = self.policy[self.hybrid[i, t]]

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
                    else:
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = self.trueState[i, t-1]
                            self.belief[i, t] = 0
                            if self.reward[i, t-1]==0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1: # is dropped back after bp
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else: # not dropped back
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                            #self.trueState[i, t] = 0  # if true world is one, pb resets it to zero
                            #self.belief[i, t] = 0
                            #self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self.policy[self.hybrid[i, t]]


    def dataGenerate_sfm(self, beliefInitial, rewInitial):
        """
        This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
        as a separate class, since at that time, the oneboxMDP class was not defined.
        In this file, all the functions are implemented under a single class.

        :return: the observations
        """

        beta = self.parameters[0]  # available food dropped back into box after button press
        gamma = self.parameters[1]  # reward becomes available
        epsilon = self.parameters[2]  # available food disappears
        rho = self.parameters[3]  # food in mouth is consumed

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]

        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)

                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
                    self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                            # action is based on softmax policy
                else:
                    if self.action[i, t-1] != pb:
                        stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
                        self.hybrid[i, t] = np.argmax(stattemp)
                            # not pressing button, hybrid state evolves probabilistically
                        self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
                    else:   # press button
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = self.trueState[i, t-1]
                            self.belief[i, t] = 0
                            if self.reward[i, t-1]==0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                                        # With probability 1- rho, reward is 1, not consumed
                                        # with probability rho, reward is 0, consumed
                        # if true world is one, pb resets it to zero with probability
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1: # is dropped back after bp
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else: # not dropped back
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])



    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)




class oneboxMDPder(oneboxMDP):
    """
    derivative of different functions with respect to the parameters
    """

    def __init__(self, discount, nq, nr, na, parameters):
        oneboxMDP.__init__(self, discount, nq, nr, na, parameters)

        self.setupMDP()
        self.solveMDP_sfm()

    def transitionDerivative(self):
        """
        calcualte the derivative of the transition probability with respect to the parameters
        :return: derivatives
        """
        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]
        Reward = 1

        dThAdepsilon = np.zeros(self.ThA.shape)
        dThAdgamma = np.zeros(self.ThA.shape)
        dThAdbeta = np.zeros(self.ThA.shape)
        dThAdrho = np.zeros(self.ThA.shape)

        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        Tb = beliefTransitionMatrixGaussian(gamma, epsilon, self.nq, sigmaTb)
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq
        Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                              np.zeros((self.nq - 2, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                              np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                              np.zeros(((self.nq - 2), 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)


        ##########################################################################
        ######## first allow some usual time, then button press ##################
        ##########################################################################
        # derivative of the belief-reward transition dynamic with respect to gamma and epsilon
        dTbdgamma, dTbdepsilon = beliefTransitionMatrixGaussianDerivative(gamma, epsilon, self.nq, sigmaTb)
        dThAdgamma[a0, :, :] = kronn(Tr, dTbdgamma)
        dThAdgamma[pb, :, :] = np.dot(Trb, dThAdgamma[a0, :, :])
        for i in range(self.na):
            dThAdgamma[i, :, :] = dThAdgamma[i, :, :].T

        dThAdepsilon[a0, :, :] = kronn(Tr, dTbdepsilon)
        dThAdepsilon[pb, :, :] = np.dot(Trb, dThAdepsilon[a0, :, :])
        for i in range(self.na):
            dThAdepsilon[i, :, :] = dThAdepsilon[i, :, :].T

        # derivative with respect to rho (appears only in Tr)
        dTrdrho = np.array([[0, 1], [0, -1]])
        dThAdrho[a0, :, :] = kronn(dTrdrho, Tb)
        dThAdrho[pb, :, :] = Trb.dot(dThAdrho[a0, :, :])
        for i in range(self.na):
            dThAdrho[i, :, :] = dThAdrho[i, :, :].T

        # derivative with respect to beta (appears only in Trb with button presing)
        dTrbdbeta = np.concatenate((np.zeros((self.nq - 1, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, 1 * bL)]),
                              np.array([np.insert([(-1) * bL], self.nq, np.zeros(self.nq))]),
                              np.zeros(((self.nq - 1), 2 * self.nq))), axis=0)
        dThAdbeta[a0, :, :] = np.zeros(self.ThA[a0, :, :].shape)
        dThAdbeta[pb, :, :] = dTrbdbeta.dot(self.ThA[a0, :, :].T).T

        '''
        ##########################################################################
        ######## only button press ##################
        ##########################################################################
        # derivative of the belief-reward transition dynamic with respect to gamma and epsilon
        dTbdgamma, dTbdepsilon = beliefTransitionMatrixGaussianDerivative(gamma, epsilon, self.nq, sigmaTb)
        dThAdgamma[a0, :, :] = kronn(Tr, dTbdgamma)
        dThAdgamma[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
        #dThAdgamma[pb, :, :] = np.dot(Trb, dThAdgamma[a0, :, :])
        for i in range(self.na):
            dThAdgamma[i, :, :] = dThAdgamma[i, :, :].T

        dThAdepsilon[a0, :, :] = kronn(Tr, dTbdepsilon)
        dThAdepsilon[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
        #dThAdepsilon[pb, :, :] = np.dot(Trb, dThAdepsilon[a0, :, :])
        for i in range(self.na):
            dThAdepsilon[i, :, :] = dThAdepsilon[i, :, :].T

        # derivative with respect to rho (appears only in Tr)
        dTrdrho = np.array([[0, 1], [0, -1]])
        dThAdrho[a0, :, :] = kronn(dTrdrho, Tb)
        dThAdrho[pb, :, :] = np.zeros(self.ThA[pb, :, :].shape)
        #dThAdrho[pb, :, :] = Trb.dot(dThAdrho[a0, :, :])
        for i in range(self.na):
            dThAdrho[i, :, :] = dThAdrho[i, :, :].T

        # derivative with respect to beta (appears only in Trb with button presing)
        dTrbdbeta = np.concatenate((np.zeros((self.nq - 1, 2 * self.nq)),
                                    np.array([np.insert([np.zeros(self.nq)], 0, 1 * bL)]),
                                    np.array([np.insert([(-1) * bL], self.nq, np.zeros(self.nq))]),
                                    np.zeros(((self.nq - 1), 2 * self.nq))), axis=0)
        dThAdbeta[a0, :, :] = np.zeros(self.ThA[a0, :, :].shape)
        dThAdbeta[pb, :, :] = dTrbdbeta.T
        #dThAdbeta[pb, :, :] = dTrbdbeta.dot(self.ThA[a0, :, :].T).T
                            # When calculating this derivative, the transpose on ThA has already been made
                            # thus need to transpose back first, and then multiply; finally transpose back again
        '''

        return dThAdbeta, dThAdgamma, dThAdepsilon, dThAdrho

    def _dpolicydQ(self):
        """
        derivative of the softmax policy with respect to Q value
        :return: dpdQ, shape: S * S
        """
        S = self.Qsfm.size
        Qexp = np.exp(self.Qsfm / temperatureQ)  # shapa: na * n

        Qexpstack = np.reshape(Qexp, S)
        Qexp_diag = np.diag(Qexpstack)

        Qexp_suminv = np.ones(self.n) / np.sum(Qexp, axis = 0)
        dpdQ = 1 / temperatureQ * np.diag(np.tile(Qexp_suminv, self.na)).dot(Qexp_diag) - \
               1 / temperatureQ * Qexp_diag.dot(np.tile(np.diag(Qexp_suminv ** 2),
                                                    (self.na, self.na))).dot(Qexp_diag)

        return dpdQ

    def dQdpara(self):
        '''
        This function is does not give the exact gradient, since the Q value
        is not convergencet, and Q iteration equation is not exactly equal
        :return:
        '''
        S = self.Qsfm.size
        Qstack = np.reshape(self.Qsfm, S)
        softpolicystack = np.reshape(self.softpolicy, S)

        Rstack = np.reshape(self.R[:, 0, :], S)

        softpolicydiag = np.diag(softpolicystack)    # policy, diagonal matrix
        ThAstack = np.reshape(np.tile(self.ThA, (1, self.na)), (self.n * self.na, self.n * self.na))
                   # stack transition probability
        Qdiag = np.diag(Qstack)
        dpdQ = self._dpolicydQ()

        # gradient of Q with respect to r
        constant_r = np.concatenate((np.zeros(self.n), - 1 * np.ones(self.n)), axis = 0)  # k2, vector
        dQdpara_r = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_r)


        dThAdbeta, dThAdgamma, dThAdepsilon, dThAdrho = self.transitionDerivative()

        # gradient of Q with respect to beta
        dThAblock_beta = block_diag(dThAdbeta[a0], dThAdbeta[pb])  # used to calculate k1
        constant_beta = dThAblock_beta.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_beta = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_beta)

        # gradient of Q with respect to gamma
        dThAblock_gamma = block_diag(dThAdgamma[a0], dThAdgamma[pb])  # used to calculate k1
        constant_gamma = dThAblock_gamma.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_gamma = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_gamma)


        # gradient of Q with respect to epsilon
        dThAblock_epsilon = block_diag(dThAdepsilon[a0], dThAdepsilon[pb])  # used to calculate k1
        constant_epsilon = dThAblock_epsilon.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_epsilon = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_epsilon)

        # gradient of Q with respect to rho
        dThAblock_rho = block_diag(dThAdrho[a0], dThAdrho[pb])  # used to calculate k1
        constant_rho = dThAblock_rho.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_rho = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_rho)

        return dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r

    def dQdpara_sim(self):
        delta = 10 ** -6

        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]

        parameters1 = [beta, gamma, epsilon, rho, pushButtonCost+delta]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        dQdpara_r = (one1.Qsfm - self.Qsfm) / delta
        dQdpara_r = np.reshape(dQdpara_r, dQdpara_r.size)

        parameters1 = [beta+delta, gamma, epsilon, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        dQdpara_beta = (one1.Qsfm - self.Qsfm) / delta
        dQdpara_beta = np.reshape(dQdpara_beta, dQdpara_beta.size)

        parameters1 = [beta, gamma+delta, epsilon, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        dQdpara_gamma = (one1.Qsfm - self.Qsfm) / delta
        dQdpara_gamma = np.reshape(dQdpara_gamma, dQdpara_gamma.size)

        parameters1 = [beta, gamma, epsilon + delta, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        dQdpara_epsilon = (one1.Qsfm - self.Qsfm) / delta
        dQdpara_epsilon = np.reshape(dQdpara_epsilon, dQdpara_epsilon.size)

        parameters1 = [beta, gamma, epsilon, rho + delta, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, parameters1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        dQdpara_rho = (one1.Qsfm - self.Qsfm) / delta
        dQdpara_rho = np.reshape(dQdpara_rho, dQdpara_rho.size)

        return dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r

    def dpdpara(self):

        # Derivative of the softmax policy with respect to the parameters

        dpdQ = self._dpolicydQ()
        #dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r= self.dQdpara()
        dQdpara_beta, dQdpara_gamma, dQdpara_epsilon, dQdpara_rho, dQdpara_r = self.dQdpara_sim()

        dpdpara_r = dpdQ.dot(dQdpara_r)
        dpdpara_beta = dpdQ.dot(dQdpara_beta)
        dpdpara_gamma = dpdQ.dot(dQdpara_gamma)
        dpdpara_epsilon = dpdQ.dot(dQdpara_epsilon)
        dpdpara_rho = dpdQ.dot(dQdpara_rho)

        dpdpara_r = np.reshape(dpdpara_r, (self.na, self.n))
        dpdpara_beta = np.reshape(dpdpara_beta, (self.na, self.n))
        dpdpara_gamma = np.reshape(dpdpara_gamma, (self.na, self.n))
        dpdpara_epsilon = np.reshape(dpdpara_epsilon, (self.na, self.n))
        dpdpara_rho = np.reshape(dpdpara_rho, (self.na, self.n))

        return dpdpara_beta, dpdpara_gamma, dpdpara_epsilon, dpdpara_rho, dpdpara_r

    def dQauxdpara_sim(self, obs, para_new):

        # Derivative of the Q auxiliary function with respect to the parameters
        # Calculated numerically by perturbing the parameters

        pi = np.ones(self.nq) / self.nq
        oneboxHMM = HMMonebox(self.ThA, self.softpolicy, pi)  #old parameter to calculate alpha, beta, gamma, xi

        delta = 10 ** -6

        beta = para_new[0]   # available food dropped back into box after button press
        gamma = para_new[1]    # reward becomes available
        epsilon = para_new[2]   # available food disappears
        rho = para_new[3]    # food in mouth is consumed
        pushButtonCost = para_new[4]

        onebox_new = oneboxMDP(self.discount, self.nq, self.nr, self.na, para_new)
        onebox_new.setupMDP()
        onebox_new.solveMDP_sfm()
        Qaux = oneboxHMM.computeQaux(obs, onebox_new.ThA, onebox_new.softpolicy)

        para1 = [beta + delta, gamma, epsilon, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
        dQauxdpara_beta = (Qaux1 - Qaux) / delta

        para1 = [beta, gamma + delta, epsilon, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
        dQauxdpara_gamma = (Qaux1 - Qaux) / delta

        para1 = [beta, gamma, epsilon + delta, rho, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
        dQauxdpara_epsilon = (Qaux1 - Qaux) / delta

        para1 = [beta, gamma, epsilon, rho + delta, pushButtonCost]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
        dQauxdpara_rho = (Qaux1 - Qaux) / delta

        para1 = [beta, gamma, epsilon, rho, pushButtonCost + delta]
        one1 = oneboxMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxHMM.computeQaux(obs, one1.ThA, one1.softpolicy)
        dQauxdpara_r = (Qaux1 - Qaux) / delta

        return dQauxdpara_beta, dQauxdpara_gamma, dQauxdpara_epsilon, dQauxdpara_rho, dQauxdpara_r


    def dQauxdpara(self, obs, para_new):

        # Derivative of the Q auxiliary function with respect to the parameters
        # Calculated analytically

        pi = np.ones(self.nq) / self.nq
        oneboxHMM = HMMonebox(self.ThA, self.softpolicy, pi)

        onebox_newde = oneboxMDPder(self.discount, self.nq, self.nr, self.na, para_new)

        dQauxdpara_beta = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
                                                  onebox_newde.transitionDerivative()[0], onebox_newde.dpdpara()[0])
        dQauxdpara_gamma = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
                                                   onebox_newde.transitionDerivative()[1], onebox_newde.dpdpara()[1])
        dQauxdpara_epsilon = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
                                                     onebox_newde.transitionDerivative()[2], onebox_newde.dpdpara()[2])
        dQauxdpara_rho = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
                                                 onebox_newde.transitionDerivative()[3], onebox_newde.dpdpara()[3])
        dQauxdpara_r = oneboxHMM.computeQauxDE(obs, onebox_newde.ThA, onebox_newde.softpolicy,
                                               np.zeros(onebox_newde.ThA.shape), onebox_newde.dpdpara()[4])

        return dQauxdpara_beta, dQauxdpara_gamma, dQauxdpara_epsilon, dQauxdpara_rho, dQauxdpara_r





