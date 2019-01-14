'''
This incorporates the oneboxtask_ini and oneboxMDPsolver and oneboxGenerate into one file with oneboxMDP object

10/23/18 add color cue for observation

'''

from __future__ import division
from boxtask_func import *
from HMMoneboxCol import *
from MDPclass import *

# we need two different transition matrices, one for each of the following actions:
a0 = 0  # a0 = do nothing
pb = 1  # pb  = push button
sigmaTb = 0.01    # variance for the gaussian approximation in belief transition matrix
temperatureQ = 0.061  # temperature for soft policy based on Q value
# qmin = 0.3
# qmax = 0.6

class oneboxColMDP:
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
        self.parameters = parameters  # [beta, gamma, epsilon, rho, pushButtonCost, Ncol, qmin, qmax]
        self.ThA = np.zeros((self.na, self.n, self.n))
        self.R = np.zeros((self.na, self.n, self.n))
        #self.temperatureQ = temperatureQ
        #self.sigmaTb = sigmaTb

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
        NumCol = np.rint(self.parameters[5]).astype(int)   # number of colors
        Ncol = NumCol - 1  # max value of color
        qmin = self.parameters[6]
        qmax = self.parameters[7]


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

        self.Trans_belief_obs, self.Obs_emis_trans, self.den = beliefTransitionMatrixGaussianCol(gamma, epsilon, qmin, qmax, Ncol, self.nq)
        self.Trans_hybrid_obs = np.zeros(((NumCol, self.n , self.n )))
        for i in range(NumCol):
            self.Trans_hybrid_obs[i] = kronn(Tr, self.Trans_belief_obs[i]).T

        Trans_belief = np.sum(self.Trans_belief_obs, axis=0)
        Tb = Trans_belief / np.tile(np.sum(Trans_belief, 0), (self.nq, 1))

        #Tb = beliefTransitionMatrixGaussian(gamma, epsilon, self.nq, self.sigmaTb)
        # softened the belief transition matrix with 2-dimensional Gaussian distribution

        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(Tr, Tb)
        # kronecker product of these transition matrices

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        self.Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                              np.zeros((self.nq - 2, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                              np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                              np.zeros(((self.nq - 2), 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        self.ThA[pb, :, :] = self.Trb.dot(self.ThA[a0, :, :])
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
        self.V = vi.V
        self.Q = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.policy = np.array(vi.policy)

        #pi = mdp.ValueIteration(self.ThA, self.R, self.discount, epsilon, niterations)
        #pi.run()
        #self.Q = self._QfromV(pi)
        #self.policy = np.array(pi.policy)


    def solveMDP_sfm(self, epsilon = 10**-6, niterations = 10000, initial_value = 0):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                V: The value for different states is returned as a proxy if needed
                Q: Q value function
                   shape: (# of actions) * (# of states)
                policy: softmax policy
        """

        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        #values = vi.run(temperatureQ)
        vi.run(temperatureQ)
        self.Vsfm = vi.V
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)

        #print(self.Qsfm)
        #print vi.max_iter
        #return vi.V, values
        #return vi.V

    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].dot(ValueIteration.V)
        return Q


class oneboxColMDPdata(oneboxColMDP):
    def __init__(self, discount, nq, nr, na, parameters, parametersExp,
                 sampleTime, sampleNum):
        oneboxColMDP.__init__(self, discount, nq, nr, na, parameters)

        self.parametersExp = parametersExp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.belief = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState = np.zeros((self.sampleNum, self.sampleTime))
        self.color = np.zeros((self.sampleNum, self.sampleTime), dtype=int)   #initialization of color

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def dataGenerate_op(self, beliefInitial, rewInitial):
        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]
        Reward = 1
        NumCol = np.rint(self.parameters[5]).astype(int)
        Ncol = NumCol - 1
        qmin = self.parameters[6]
        qmax = self.parameters[7]

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]
        qmin_e = self.parametersExp[2]
        qmax_e = self.parametersExp[3]

        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)
                    q = self.trueState[i, t] * qmin_e + (1 - self.trueState[i, t]) * qmax_e
                    self.color[i, t] = np.random.binomial(Ncol, q)  # color

                    # The first belief is not based on world state
                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]  # This is for one box only
                    self.action[i, t] = self.policy[self.hybrid[i, t]]  # action is based on optimal policy
                else:
                    if self.action[i, t - 1] != pb:
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)

                        q = self.trueState[i, t] * qmin_e + (1 - self.trueState[i, t]) * qmax_e
                        self.color[i, t] = np.random.binomial(Ncol, q)  # color

                        self.belief[i, t] = np.argmax(
                            np.random.multinomial(1, self.den[self.color[i, t], :, self.belief[i, t - 1]], size=1))

                        if self.reward[i, t - 1] == 0:
                            self.reward[i, t] = 0
                        else:
                            self.reward[i, t] = np.random.binomial(1, 1 - rho)

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]  # This is for one box only
                        self.action[i, t] = self.policy[self.hybrid[i, t]]

                    else:
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = 0
                            self.color[i, t] = Ncol
                            self.belief[i, t] = 0
                            if self.reward[i, t - 1] == 0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1:  # is dropped back after bp
                                self.color[i, t] = 0
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else:  # not dropped back
                                self.color[i, t] = Ncol
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self.policy[self.hybrid[i, t]]


    def dataGenerate_sfm(self, beliefInitial, rewInitial):
        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]
        Reward = 1
        NumCol = np.rint(self.parameters[5]).astype(int)
        Ncol = NumCol - 1
        qmin = self.parameters[6]
        qmax = self.parameters[7]

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]
        qmin_e = self.parametersExp[2]
        qmax_e = self.parametersExp[3]

        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)
                    q = self.trueState[i, t] * qmin_e + (1 - self.trueState[i, t]) * qmax_e
                    self.color[i, t] = np.random.binomial(Ncol, q)  # color

                    # The first belief is not based on world state
                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]  # This is for one box only
                    self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                    # action is based on optimal policy
                else:
                    if self.action[i, t - 1] != pb:
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)

                        q = self.trueState[i, t] * qmin_e + (1 - self.trueState[i, t]) * qmax_e
                        self.color[i, t] = np.random.binomial(Ncol, q)  # color

                        self.belief[i, t] = np.argmax(
                            np.random.multinomial(1, self.den[self.color[i, t], :, self.belief[i, t - 1]], size=1))

                        if self.reward[i, t - 1] == 0:
                            self.reward[i, t] = 0
                        else:
                            self.reward[i, t] = np.random.binomial(1, 1 - rho)

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]  # This is for one box only
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])

                    else:
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = 0
                            self.color[i, t] = Ncol
                            self.belief[i, t] = 0
                            if self.reward[i, t - 1] == 0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1:  # is dropped back after bp
                                self.color[i, t] = 0
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else:  # not dropped back
                                self.color[i, t] = Ncol
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])

    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)


class oneboxColMDPder(oneboxColMDP):
    """
    derivative of different functions with respect to the parameters
    """

    def __init__(self, discount, nq, nr, na, parameters):
        oneboxColMDP.__init__(self, discount, nq, nr, na, parameters)

        self.setupMDP()
        self.solveMDP_sfm()


    def dQauxdpara_sim(self, obs, para_new):

        # Derivative of the Q auxiliary function with respect to the parameters
        # Calculated numerically by perturbing the parameters

        pi = np.ones(self.nq) / self.nq
        oneboxColHMM = HMMoneboxCol(self.ThA, self.softpolicy, self.Trans_hybrid_obs, self.Obs_emis_trans,
                                    pi, Ncol = np.rint(self.parameters[5]).astype(int) - 1)  #old parameter to calculate alpha, beta, gamma, xi

        delta = 10 ** -6

        beta = para_new[0]   # available food dropped back into box after button press
        gamma = para_new[1]    # reward becomes available
        epsilon = para_new[2]   # available food disappears
        rho = para_new[3]    # food in mouth is consumed
        pushButtonCost = para_new[4]
        NumCol = para_new[5]
        Ncol = NumCol - 1
        qmin = para_new[6]
        qmax = para_new[7]

        onebox_new = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para_new)
        onebox_new.setupMDP()
        onebox_new.solveMDP_sfm()
        Qaux = oneboxColHMM.computeQaux(obs, onebox_new.ThA, onebox_new.softpolicy,
                                        onebox_new.Trans_hybrid_obs, onebox_new.Obs_emis_trans)

        #delta
        para1 = [beta + delta, gamma, epsilon, rho, pushButtonCost, NumCol, qmin, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_beta = (Qaux1 - Qaux) / delta

        # gamma
        para1 = [beta , gamma + delta, epsilon, rho, pushButtonCost, NumCol, qmin, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_gamma = (Qaux1 - Qaux) / delta

        # epsilon
        para1 = [beta , gamma , epsilon + delta , rho, pushButtonCost, NumCol, qmin, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_epsilon = (Qaux1 - Qaux) / delta

        #rho
        para1 = [beta , gamma , epsilon, rho + delta, pushButtonCost, NumCol, qmin, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_rho = (Qaux1 - Qaux) / delta

        # pb cost
        para1 = [beta , gamma , epsilon, rho, pushButtonCost + delta, NumCol, qmin, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_pb = (Qaux1 - Qaux) / delta

        # qmin
        para1 = [beta , gamma , epsilon, rho, pushButtonCost , NumCol, qmin + delta, qmax]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_qmin = (Qaux1 - Qaux) / delta

        # qmax
        para1 = [beta , gamma , epsilon, rho, pushButtonCost , NumCol, qmin , qmax + delta]
        one1 = oneboxColMDP(self.discount, self.nq, self.nr, self.na, para1)
        one1.setupMDP()
        one1.solveMDP_sfm()
        Qaux1 = oneboxColHMM.computeQaux(obs, one1.ThA, one1.softpolicy,
                                         one1.Trans_hybrid_obs, one1.Obs_emis_trans)
        dQauxdpara_qmax = (Qaux1 - Qaux) / delta

        return dQauxdpara_beta, dQauxdpara_gamma, dQauxdpara_epsilon, dQauxdpara_rho, dQauxdpara_pb, 0, dQauxdpara_qmin, dQauxdpara_qmax
        #return 0, dQauxdpara_gamma, dQauxdpara_epsilon, 0, dQauxdpara_pb, 0, dQauxdpara_qmin, dQauxdpara_qmax

