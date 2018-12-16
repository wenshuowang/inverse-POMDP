'''
This incorporates the twoboxtask_ini and twoboxMDPsolver and twoboxGenerate into one file with twoboxMDP object
analogy to onebox.py
'''

from __future__ import division
from boxtask_func import *
from HMMtwobox import *
from MDPclass import *

import numpy.matlib
from scipy.linalg import block_diag
from numpy.linalg import inv


#from twoboxtask_ini import *

# we need five different transition matrices, one for each of the following actions:
a0 = 0    # a0 = do nothing
g0 = 1    # g0 = go to location 0
g1 = 2    # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3    # g2 = go toward box 2 (via location 0 if from 1)
pb = 4    # pb  = push button
sigmaTb = 0.01  #0.1    # variance for the gaussian approximation in belief transition matrix
temperatureQ = 0.2  #(0.1 for Neda, others usually 0.2) #0.2  # temperature for soft policy based on Q value

class twoboxMDP:
    def __init__(self, discount, nq, nr, na, nl, parameters):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.nl = nl   # number of locations
        self.n = self.nq * self.nq * self.nr * self.nl   # total number of states
        self.parameters = parameters  # [beta, gamma, epsilon, rho]
        self.ThA = np.zeros((self.na, self.n, self.n))
        self.R = np.zeros((self.na, self.n, self.n))

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'twoboxtask_ini.py'
        :return:
                ThA: transition probability,
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                    shape: (# of action) * (# of states, old state) * (# of states, new state)
        """
        # beta = self.parameters[0]     # available food dropped back into box after button press
        # gamma1 = self.parameters[1]   # reward becomes available in box 1
        # gamma2 = self.parameters[2]   # reward becomes available in box 2
        # delta = self.parameters[3]    # animal trips, doesn't go to target location
        # direct = self.parameters[4]   # animal goes right to target, skipping location 0
        # epsilon1 = self.parameters[5] # available food disappears from box 1
        # epsilon2 = self.parameters[6] # available food disappears from box 2
        # rho = self.parameters[7]      # food in mouth is consumed
        # # eta = .0001                 # random diffusion of belief
        #
        # # State rewards
        # Reward = self.parameters[8]   # reward per time step with food in mouth
        # Groom = self.parameters[9]     # location 0 reward
        #
        # # Action costs
        # travelCost = self.parameters[10]
        # pushButtonCost = self.parameters[11]


        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        # initialize probability distribution over states (belief and world)
        pr0 = np.array([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pl0 = np.array([1, 0, 0])  # (l=0, l=1, l=2) initial location is at L=0
        pb10 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)
        pb20 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)

        ph0 = kronn(pl0, pb10, pr0, pb20)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        # Tb1 = beliefTransitionMatrix(gamma1, epsilon1, nq, eta)
        # Tb2 = beliefTransitionMatrix(gamma2, epsilon2, nq, eta)
        Tb1 = beliefTransitionMatrixGaussian(gamma1, epsilon1, self.nq, sigmaTb)
        Tb2 = beliefTransitionMatrixGaussian(gamma2, epsilon2, self.nq, sigmaTb)

        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(np.identity(self.nl), Tb1, Tr, Tb2)
        # kronecker product of these transition matrices


        # ACTION: go to location 0/1/2
        Tl0 = np.array(
            [[1, 1 - delta, 1 - delta], [0, delta, 0], [0, 0, delta]])  # go to loc 0 (with error of delta)
        Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        [0, 0, delta]])  # go to box 1 (with error of delta)
        Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
        self.ThA[g0, :, :] = kronn(Tl0, Tb1, Tr, Tb2)
        self.ThA[g1, :, :] = kronn(Tl1, Tb1, Tr, Tb2)
        self.ThA[g2, :, :] = kronn(Tl2, Tb1, Tr, Tb2)

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        Trb2 = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                               np.zeros((self.nq - 2, 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                               np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                               np.zeros(((self.nq - 2), 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        Tb1r = reversekron(Trb2, np.array([2, self.nq]))
        Th = block_diag(np.identity(self.nq * self.nr * self.nq),
                        np.kron(Tb1r, np.identity(self.nq)),
                        np.kron(np.identity(self.nq), Trb2))
        self.ThA[pb, :, :] = Th.dot(self.ThA[a0, :, :])
        # self.ThA[pb, :, :] = Th

        Reward_h = tensorsumm(np.array([[Groom, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, Reward]]),
                              np.zeros((1, self.nq)))
        Reward_a = - np.array([0, travelCost, travelCost, travelCost, pushButtonCost])

        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3
        # R = Reward[:, 0, :].T
        self.R = Reward

        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T


    def solveMDP_op(self, epsilon = 0.001, niterations = 10000, initial_value=0):
        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        # optimal policy, stopping criterion changed to "converged Qvalue"
        vi.run()
        self.Q = self._QfromV(vi)
        self.policy = np.array(vi.policy)
        self.Vop = vi.V

    def solveMDP_sfm(self, epsilon = 0.001, niterations = 10000, initial_value=0):

        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        self.Vsfm = vi.V

    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].dot(ValueIteration.V)
        return Q


class twoboxMDPdata(twoboxMDP):
    def __init__(self, discount, nq, nr, na, nl, parameters, sampleTime, sampleNum):
        twoboxMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action, assumed to be optimal
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.location = np.empty((sampleNum, sampleTime), int)  # initialize location state
        self.belief1 = np.empty((self.sampleNum, self.sampleTime), int)
        self.belief2 = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()


    def dataGenerate_op(self, belief1Initial, rewInitial, belief2Initial, locationInitial):
        self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))

        ## Parameters
        # beta = self.parameters[0]     # available food dropped back into box after button press
        # gamma1 = self.parameters[1]   # reward becomes available in box 1
        # gamma2 = self.parameters[2]   # reward becomes available in box 2
        # delta = self.parameters[3]    # animal trips, doesn't go to target location
        # direct = self.parameters[4]   # animal goes right to target, skipping location 0
        # epsilon1 = self.parameters[5] # available food disappears from box 1
        # epsilon2 = self.parameters[6] # available food disappears from box 2
        # rho = self.parameters[7]      # food in mouth is consumed
        # # eta = .0001                 # random diffusion of belief
        # # State rewards
        # Reward = self.parameters[8]   # reward per time step with food in mouth
        # Groom = self.parameters[9]     # location 0 reward
        # # Action costs
        # travelCost = self.parameters[10]
        # pushButtonCost = self.parameters[11]

        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        Tb1 = beliefTransitionMatrixGaussian(gamma1, epsilon1, self.nq, sigmaTb)
        Tb2 = beliefTransitionMatrixGaussian(gamma2, epsilon2, self.nq, sigmaTb)

        ## Generate data
        for n in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    self.trueState1[n, t] = np.random.binomial(1, gamma1)
                    self.trueState2[n, t] = np.random.binomial(1, gamma2)

                    self.location[n, t], self.belief1[n, t], self.reward[n, t], self.belief2[
                        n, t] = locationInitial, belief1Initial, rewInitial, belief2Initial
                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                        self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.action[n, t] = self.policy[self.hybrid[n, t]]
                else:
                    # variables evolve with dynamics
                    if self.action[n, t - 1] != pb:
                        acttemp = np.random.multinomial(1, self.ThA[self.action[n, t - 1], self.hybrid[n, t - 1], :], size=1)
                        self.hybrid[n, t] = np.argmax(acttemp)

                        self.location[n, t] = divmod(self.hybrid[n, t], self.nq * self.nr * self.nq)[0]
                        self.belief1[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq), self.nr * self.nq)[0]
                        self.reward[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) \
                                                   - self.belief1[n, t] * (self.nr * self.nq), self.nq)[0]
                        self.belief2[n, t] = self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) - \
                                             self.belief1[n, t] * (self.nr * self.nq) - self.reward[n, t] * self.nq

                        self.action[n, t] = self.policy[self.hybrid[n, t]]

                        # button not pressed, then true world dynamic is not affected by actions
                        if self.trueState1[n, t - 1] == 0:
                            self.trueState1[n, t] = np.random.binomial(1, gamma1)
                        else:
                            self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)

                        if self.trueState2[n, t - 1] == 0:
                            self.trueState2[n, t] = np.random.binomial(1, gamma2)
                        else:
                            self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)

                    if self.action[n, t - 1] == pb:  # press button
                        self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location

                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState1[n, t - 1] == 0:
                            self.trueState1[n, t - 1] = np.random.binomial(1, gamma1)
                        else:
                            self.trueState1[n, t - 1] = 1 - np.random.binomial(1, epsilon1)

                        if self.trueState2[n, t - 1] == 0:
                            self.trueState2[n, t - 1] = np.random.binomial(1, gamma2)
                        else:
                            self.trueState2[n, t - 1] = 1 - np.random.binomial(1, epsilon2)
                        #### for pb action, wait for usual time and then pb  #############


                        if self.location[n, t] == 1:  # consider location 1 case
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, Tb2[:, self.belief2[n, t - 1]], size=1))
                            # belief on box 2 is independent on box 1
                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = np.random.binomial(1, gamma2)
                            else:
                                self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)

                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = self.trueState1[n, t - 1]
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief1[n, t] = 0  # after open the box, the animal is sure that there is no food there
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.trueState1[n, t] = 0  # if true world is one, pb resets it to zero
                                self.belief1[n, t] = 0
                                self.reward[n, t] = 1  # give some reward

                        if self.location[n, t] == 2:  # consider location 2 case
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, Tb1[:, self.belief1[n, t - 1]], size=1))
                            # belief on box 1 is independent on box 2
                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = np.random.binomial(1, gamma1)
                            else:
                                self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)

                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = self.trueState2[n, t - 1]
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief2[n, t] = 0  # after open the box, the animal is sure that there is no food there
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.trueState2[n, t] = 0  # if true world is one, pb resets it to zero
                                self.belief2[n, t] = 0
                                self.reward[n, t] = 1  # give some reward

                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.action[n, t] = self.policy[self.hybrid[n, t]]



    def dataGenerate_sfm(self, belief1Initial, rewInitial, belief2Initial, locationInitial):
        self.trueState1 = np.zeros((self.sampleNum, self.sampleTime))
        self.trueState2 = np.zeros((self.sampleNum, self.sampleTime))

        ## Parameters
        # beta = self.parameters[0]     # available food dropped back into box after button press
        # gamma1 = self.parameters[1]   # reward becomes available in box 1
        # gamma2 = self.parameters[2]   # reward becomes available in box 2
        # delta = self.parameters[3]    # animal trips, doesn't go to target location
        # direct = self.parameters[4]   # animal goes right to target, skipping location 0
        # epsilon1 = self.parameters[5] # available food disappears from box 1
        # epsilon2 = self.parameters[6] # available food disappears from box 2
        # rho = self.parameters[7]      # food in mouth is consumed
        # # eta = .0001                 # random diffusion of belief
        # # State rewards
        # Reward = self.parameters[8]   # reward per time step with food in mouth
        # Groom = self.parameters[9]     # location 0 reward
        # # Action costs
        # travelCost = self.parameters[10]
        # pushButtonCost = self.parameters[11]

        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        Tb1 = beliefTransitionMatrixGaussian(gamma1, epsilon1, self.nq, sigmaTb)
        Tb2 = beliefTransitionMatrixGaussian(gamma2, epsilon2, self.nq, sigmaTb)

        ## Generate data
        for n in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    # Initialize the true world states, sensory information and latent states
                    self.trueState1[n, t] = np.random.binomial(1, gamma1)
                    self.trueState2[n, t] = np.random.binomial(1, gamma2)

                    self.location[n, t], self.belief1[n, t], self.reward[n, t], self.belief2[
                        n, t] = locationInitial, belief1Initial, rewInitial, belief2Initial
                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) + \
                                        self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])
                else:
                    if self.action[n, t - 1] == pb and self.location[n, t - 1] == 0:
                        self.action[n, t - 1] = a0  # cannot press button at location 0

                    # variables evolve with dynamics
                    if self.action[n, t - 1] != pb:
                        acttemp = np.random.multinomial(1, self.ThA[self.action[n, t - 1], self.hybrid[n, t - 1], :], size=1)
                        #print acttemp
                        self.hybrid[n, t] = np.argmax(acttemp)

                        self.location[n, t] = divmod(self.hybrid[n, t], self.nq * self.nr * self.nq)[0]
                        self.belief1[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq), self.nr * self.nq)[0]
                        self.reward[n, t] = divmod(self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) \
                                                   - self.belief1[n, t] * (self.nr * self.nq), self.nq)[0]
                        self.belief2[n, t] = self.hybrid[n, t] - self.location[n, t] * (self.nq * self.nr * self.nq) - \
                                             self.belief1[n, t] * (self.nr * self.nq) - self.reward[n, t] * self.nq

                        self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])

                        # button not pressed, then true world dynamic is not affected by actions
                        if self.trueState1[n, t - 1] == 0:
                            self.trueState1[n, t] = np.random.binomial(1, gamma1)
                        else:
                            self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)

                        if self.trueState2[n, t - 1] == 0:
                            self.trueState2[n, t] = np.random.binomial(1, gamma2)
                        else:
                            self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)

                    if self.action[n, t - 1] == pb:  # press button
                        self.location[n, t] = self.location[n, t - 1]  # pressing button does not change location

                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState1[n, t - 1] == 0:
                            self.trueState1[n, t - 1] = np.random.binomial(1, gamma1)
                        else:
                            self.trueState1[n, t - 1] = 1 - np.random.binomial(1, epsilon1)

                        if self.trueState2[n, t - 1] == 0:
                            self.trueState2[n, t - 1] = np.random.binomial(1, gamma2)
                        else:
                            self.trueState2[n, t - 1] = 1 - np.random.binomial(1, epsilon2)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.location[n, t] == 1:  # consider location 1 case
                            self.belief2[n, t] = np.argmax(np.random.multinomial(1, Tb2[:, self.belief2[n, t - 1]], size=1))
                            # belief on box 2 is independent on box 1
                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = np.random.binomial(1, gamma2)
                            else:
                                self.trueState2[n, t] = 1 - np.random.binomial(1, epsilon2)

                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = self.trueState1[n, t - 1]
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief1[n, t] = 0  # after open the box, the animal is sure that there is no food there
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.trueState1[n, t] = 0  # if true world is one, pb resets it to zero
                                self.belief1[n, t] = 0
                                self.reward[n, t] = 1  # give some reward

                        if self.location[n, t] == 2:  # consider location 2 case
                            self.belief1[n, t] = np.argmax(np.random.multinomial(1, Tb1[:, self.belief1[n, t - 1]], size=1))
                            # belief on box 1 is independent on box 2
                            if self.trueState1[n, t - 1] == 0:
                                self.trueState1[n, t] = np.random.binomial(1, gamma1)
                            else:
                                self.trueState1[n, t] = 1 - np.random.binomial(1, epsilon1)

                            if self.trueState2[n, t - 1] == 0:
                                self.trueState2[n, t] = self.trueState2[n, t - 1]
                                # if true world is zero, pb does not change real state
                                # assume that the real state does not change during button press
                                self.belief2[n, t] = 0  # after open the box, the animal is sure that there is no food there
                                if self.reward[n, t - 1] == 0:  # reward depends on previous time frame
                                    self.reward[n, t] = 0
                                else:
                                    self.reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                            else:
                                self.trueState2[n, t] = 0  # if true world is one, pb resets it to zero
                                self.belief2[n, t] = 0
                                self.reward[n, t] = 1  # give some reward

                    self.hybrid[n, t] = self.location[n, t] * (self.nq * self.nr * self.nq) + self.belief1[n, t] * (self.nr * self.nq) \
                                        + self.reward[n, t] * self.nq + self.belief2[n, t]  # hybrid state, for policy choosing
                    #print self.location[n, t], self.belief1[n, t], self.reward[n, t], self.belief2[n, t], self.hybrid[n, t], self.action[n, t-1]
                    self.action[n, t] = self._chooseAction(self.softpolicy.T[self.hybrid[n, t]])


    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)


class twoboxMDPder(twoboxMDP):
    """
    Derivatives of different functions with respect to the parameters
    """
    def __init__(self, discount, nq, nr, na, nl, parameters, initial_valueV = 0):
        twoboxMDP.__init__(self, discount, nq, nr, na, nl, parameters)

        self.setupMDP()
        self.solveMDP_sfm(initial_value = initial_valueV)

    def transitionDerivative(self):

        ## Parameters
        # beta = self.parameters[0]     # available food dropped back into box after button press
        # gamma1 = self.parameters[1]   # reward becomes available in box 1
        # gamma2 = self.parameters[2]   # reward becomes available in box 2
        # delta = self.parameters[3]    # animal trips, doesn't go to target location
        # direct = self.parameters[4]   # animal goes right to target, skipping location 0
        # epsilon1 = self.parameters[5] # available food disappears from box 1
        # epsilon2 = self.parameters[6] # available food disappears from box 2
        # rho = self.parameters[7]      # food in mouth is consumed
        # # eta = .0001                 # random diffusion of belief
        # # State rewards
        # Reward = self.parameters[8]   # reward per time step with food in mouth
        # Groom = self.parameters[9]     # location 0 reward
        # # Action costs
        # travelCost = self.parameters[10]
        # pushButtonCost = self.parameters[11]

        beta = 0     # available food dropped back into box after button press
        gamma1 = self.parameters[0]   # reward becomes available in box 1
        gamma2 = self.parameters[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2] # available food disappears from box 1
        epsilon2 = self.parameters[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        dThAdgamma1 = np.zeros(self.ThA.shape)
        dThAdgamma2 = np.zeros(self.ThA.shape)
        dThAdepsilon1 = np.zeros(self.ThA.shape)
        dThAdepsilon2 = np.zeros(self.ThA.shape)
        dThAdbeta = np.zeros(self.ThA.shape)
        dThAdrho = np.zeros(self.ThA.shape)

        # setup single-variable transition matrices
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        # Tb1 = beliefTransitionMatrix(gamma1, epsilon1, nq, eta)
        # Tb2 = beliefTransitionMatrix(gamma2, epsilon2, nq, eta)
        Tb1 = beliefTransitionMatrixGaussian(gamma1, epsilon1, self.nq, sigmaTb)
        Tb2 = beliefTransitionMatrixGaussian(gamma2, epsilon2, self.nq, sigmaTb)

        # ACTION: do nothing
        dTrdrho = np.array([[0, 1], [0, -1]])

        dTb1dgamma1, dTb1depsilon1 = beliefTransitionMatrixGaussianDerivative(gamma1, epsilon1, self.nq, sigmaTb)
        dTb2dgamma2, dTb2depsilon2 = beliefTransitionMatrixGaussianDerivative(gamma2, epsilon2, self.nq, sigmaTb)

        dThAdgamma1[a0, :, :] = kronn(np.identity(self.nl), dTb1dgamma1, Tr, Tb2)
        dThAdepsilon1[a0, :, :] = kronn(np.identity(self.nl), dTb1depsilon1, Tr, Tb2)
        dThAdgamma2[a0, :, :] = kronn(np.identity(self.nl), Tb1, Tr, dTb2dgamma2)
        dThAdepsilon2[a0, :, :] = kronn(np.identity(self.nl), Tb1, Tr, dTb2depsilon2)
        dThAdrho[a0, :, :] = kronn(np.identity(self.nl), Tb1, dTrdrho, Tb2)
        dThAdbeta[a0, :, :] = np.zeros(self.ThA[a0, :, :].shape)

        # ACTION: go to location 0/1/2
        Tl0 = np.array(
            [[1, 1 - delta, 1 - delta], [0, delta, 0], [0, 0, delta]])  # go to loc 0 (with error of delta)
        Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                        [0, 0, delta]])  # go to box 1 (with error of delta)
        Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                        [1 - delta, direct, 1]])  # go to box 2 (with error of delta)

        dThAdgamma1[g0, :, :] = kronn(Tl0, dTb1dgamma1, Tr, Tb2)
        dThAdepsilon1[g0, :, :] = kronn(Tl0, dTb1depsilon1, Tr, Tb2)
        dThAdgamma2[g0, :, :] = kronn(Tl0, Tb1, Tr, dTb2dgamma2)
        dThAdepsilon2[g0, :, :] = kronn(Tl0, Tb1, Tr, dTb2depsilon2)
        dThAdrho[g0, :, :] = kronn(Tl0, Tb1, dTrdrho, Tb2)
        dThAdbeta[g0, :, :] = np.zeros(self.ThA[g0, :, :].shape)

        dThAdgamma1[g1, :, :] = kronn(Tl1, dTb1dgamma1, Tr, Tb2)
        dThAdepsilon1[g1, :, :] = kronn(Tl1, dTb1depsilon1, Tr, Tb2)
        dThAdgamma2[g1, :, :] = kronn(Tl1, Tb1, Tr, dTb2dgamma2)
        dThAdepsilon2[g1, :, :] = kronn(Tl1, Tb1, Tr, dTb2depsilon2)
        dThAdrho[g1, :, :] = kronn(Tl1, Tb1, dTrdrho, Tb2)
        dThAdbeta[g1, :, :] = np.zeros(self.ThA[g1, :, :].shape)

        dThAdgamma1[g2, :, :] = kronn(Tl2, dTb1dgamma1, Tr, Tb2)
        dThAdepsilon1[g2, :, :] = kronn(Tl2, dTb1depsilon1, Tr, Tb2)
        dThAdgamma2[g2, :, :] = kronn(Tl2, Tb1, Tr, dTb2dgamma2)
        dThAdepsilon2[g2, :, :] = kronn(Tl2, Tb1, Tr, dTb2depsilon2)
        dThAdrho[g2, :, :] = kronn(Tl2, Tb1, dTrdrho, Tb2)
        dThAdbeta[g2, :, :] = np.zeros(self.ThA[g2, :, :].shape)

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        Trb2 = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                               np.zeros((self.nq - 2, 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                               np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                               np.zeros(((self.nq - 2), 2 * self.nq)),
                               np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        Tb1r = reversekron(Trb2, np.array([2, self.nq]))
        Th = block_diag(np.identity(self.nq * self.nr * self.nq),
                        np.kron(Tb1r, np.identity(self.nq)),
                        np.kron(np.identity(self.nq), Trb2))

        dTrb2dbeta = np.concatenate((np.zeros((self.nq - 1, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, 1 * bL)]),
                              np.array([np.insert([(-1) * bL], self.nq, np.zeros(self.nq))]),
                              np.zeros(((self.nq - 1), 2 * self.nq))), axis=0)
        dTb1rdbeta = reversekron(dTrb2dbeta, np.array([2, self.nq]))
        dThdbeta = block_diag(np.identity(self.nq * self.nr * self.nq),
                        np.kron(dTb1rdbeta, np.identity(self.nq)),
                        np.kron(np.identity(self.nq), dTrb2dbeta))

        dThAdgamma1[pb, :, :] = Th.dot(dThAdgamma1[a0, :, :])
        dThAdepsilon1[pb, :, :] = Th.dot(dThAdepsilon1[a0, :, :])
        dThAdgamma2[pb, :, :] = Th.dot(dThAdgamma2[a0, :, :])
        dThAdepsilon2[pb, :, :] = Th.dot(dThAdepsilon2[a0, :, :])
        dThAdrho[pb, :, :] = Th.dot(dThAdrho[a0, :, :])
        dThAdbeta[pb, :, :] = dThdbeta.dot(self.ThA[a0, :, :].T)

        for i in range(self.na):
            dThAdgamma1[i, :, :] = dThAdgamma1[i, :, :].T
            dThAdepsilon1[i, :, :] = dThAdepsilon1[i, :, :].T
            dThAdgamma2[i, :, :] = dThAdgamma2[i, :, :].T
            dThAdepsilon2[i, :, :] = dThAdepsilon2[i, :, :].T
            dThAdrho[i, :, :] = dThAdrho[i, :, :].T
            dThAdbeta[i, :, :] = dThAdbeta[i, :, :].T

        #return dThAdgamma1, dThAdgamma2, dThAdepsilon1, dThAdepsilon2, dThAdrho, dThAdbeta
        return dThAdgamma1, dThAdgamma2, dThAdepsilon1, dThAdepsilon2

    def dpdQ(self):
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
        S = self.Qsfm.size
        Qstack = np.reshape(self.Qsfm, S)
        softpolicystack = np.reshape(self.softpolicy, S)

        Rstack = np.reshape(self.R[:, 0, :], S)

        softpolicydiag = np.diag(softpolicystack)  # policy, diagonal matrix
        ThAstack = np.reshape(np.tile(self.ThA, (1, self.na)), (self.n * self.na, self.n * self.na))
        # stack transition probability
        Qdiag = np.diag(Qstack)
        dpdQ = self.dpdQ()

        dThAdgamma1, dThAdgamma2, dThAdepsilon1, dThAdepsilon2 = self.transitionDerivative()
        ## gradient of Q with respect the the parameters (gamma 1/2, epsilon 1/2)
        #dThAblock_gamma1 = block_diag(dThAdgamma1[a0], dThAdgamma1[pb])  # used to calculate k1
        dThAblock_gamma1 = dThAdgamma1[0]
        for i in range(1, self.na): dThAblock_gamma1  = block_diag(dThAblock_gamma1 , dThAdgamma1[i])
        constant_gamma1 = dThAblock_gamma1.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                             dot(Qstack * softpolicystack) + Rstack)
        dQdpara_gamma1 = inv(np.eye(S) - self.discount * ThAstack.
                            dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_gamma1)
        #dQdpara_gamma1 = np.reshape(dQdpara_gamma1, (self.na, self.n))

        dThAblock_gamma2 = dThAdgamma2[0]
        for i in range(1, self.na): dThAblock_gamma2 = block_diag(dThAblock_gamma2, dThAdgamma2[i])
        #dThAblock_gamma2 = block_diag(dThAdgamma2[a0], dThAdgamma2[pb])  # used to calculate k1
        constant_gamma2 = dThAblock_gamma2.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                             dot(Qstack * softpolicystack) + Rstack)
        dQdpara_gamma2 = inv(np.eye(S) - self.discount * ThAstack.
                            dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_gamma2)
        #dQdpara_gamma2 = np.reshape(dQdpara_gamma2, (self.na, self.n))

        dThAblock_epsilon1 = dThAdepsilon1[0]
        for i in range(1, self.na): dThAblock_epsilon1 = block_diag(dThAblock_epsilon1, dThAdepsilon1[i])
        #dThAblock_epsilon1 = block_diag(dThAdepsilon1[a0], dThAdepsilon1[pb])  # used to calculate k1
        constant_epsilon1 = dThAblock_epsilon1.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_epsilon1 = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_epsilon1)
        #dQdpara_epsilon1 = np.reshape(dQdpara_epsilon1, (self.na, self.n))

        dThAblock_epsilon2 = dThAdepsilon2[0]
        for i in range(1, self.na): dThAblock_epsilon2 = block_diag(dThAblock_epsilon2, dThAdepsilon2[i])
        #dThAblock_epsilon2 = block_diag(dThAdepsilon2[a0], dThAdepsilon2[pb])  # used to calculate k1
        constant_epsilon2 = dThAblock_epsilon2.dot(self.discount * np.matlib.repmat(np.eye(self.n), self.na, self.na).
                                           dot(Qstack * softpolicystack) + Rstack)
        dQdpara_epsilon2 = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_epsilon2)
        #dQdpara_epsilon2 = np.reshape(dQdpara_epsilon2, (self.na, self.n))

        ## gradient of Q with respect to costs
        # Grooming
        Reward_h_der = tensorsumm(np.array([[1, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, 0]]),
                              np.zeros((1, self.nq)))
        Reward_a_der = - np.array([0, 0, 0, 0, 0])
        [R1, R2, R3] = np.meshgrid(Reward_a_der.T, Reward_h_der, Reward_h_der, indexing='ij')
        Reward_der = R1 + R3
        constant_Groom = np.hstack(np.sum(Reward_der * self.ThA, axis = 2))
        dQdpara_Groom = inv(np.eye(S) - self.discount * ThAstack.
                        dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_Groom)
        #dQdpara_Groom = np.reshape(dQdpara_Groom, (self.na, self.n))

        # travelCost
        Reward_h_der = tensorsumm(np.array([[0, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, 0]]),
                              np.zeros((1, self.nq)))
        Reward_a_der = - np.array([0, 1, 1, 1, 0])
        [R1, R2, R3] = np.meshgrid(Reward_a_der.T, Reward_h_der, Reward_h_der, indexing='ij')
        Reward_der = R1 + R3
        constant_travelCost = np.hstack(np.sum(Reward_der * self.ThA, axis=2))
        dQdpara_travelCost = inv(np.eye(S) - self.discount * ThAstack.
                            dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_travelCost)
        #dQdpara_travelCost = np.reshape(dQdpara_travelCost, (self.na, self.n))

        #pushButtonCost
        Reward_h_der = tensorsumm(np.array([[0, 0, 0]]), np.zeros((1, self.nq)), np.array([[0, 0]]),
                              np.zeros((1, self.nq)))
        Reward_a_der = - np.array([0, 0, 0, 0, 1])
        [R1, R2, R3] = np.meshgrid(Reward_a_der.T, Reward_h_der, Reward_h_der, indexing='ij')
        Reward_der = R1 + R3
        constant_pushButtonCost = np.hstack(np.sum(Reward_der * self.ThA, axis=2))
        dQdpara_pushButtonCost = inv(np.eye(S) - self.discount * ThAstack.
                                 dot(Qdiag.dot(dpdQ) + softpolicydiag)).dot(constant_pushButtonCost)
        #dQdpara_pushButtonCost = np.reshape(dQdpara_pushButtonCost, (self.na, self.n))

        return dQdpara_gamma1, dQdpara_gamma2, dQdpara_epsilon1, dQdpara_epsilon2, \
               dQdpara_Groom, dQdpara_travelCost, dQdpara_pushButtonCost

    def dQdpara_sim(self):
        perturb = 10 ** -6

        beta = 0  # available food dropped back into box after button press
        gamma1 = self.parameters[0]  # reward becomes available in box 1
        gamma2 = self.parameters[1]  # reward becomes available in box 2
        delta = 0  # animal trips, doesn't go to target location
        direct = 0  # animal goes right to target, skipping location 0
        epsilon1 = self.parameters[2]  # available food disappears from box 1
        epsilon2 = self.parameters[3]  # available food disappears from box 2
        rho = 1
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = self.parameters[4]     # location 0 reward
        # Action costs
        travelCost = self.parameters[5]
        pushButtonCost = self.parameters[6]

        para1 = [gamma1 + perturb, gamma2, epsilon1, epsilon2, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_gamma1 = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_gamma1 = np.reshape(dQdpara_gamma1, dQdpara_gamma1.size)

        para1 = [gamma1, gamma2+ perturb, epsilon1, epsilon2, Groom ,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_gamma2 = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_gamma2 = np.reshape(dQdpara_gamma2, dQdpara_gamma2.size)

        para1 = [gamma1, gamma2, epsilon1 + perturb, epsilon2, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_epsilon1 = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_epsilon1 = np.reshape(dQdpara_epsilon1, dQdpara_epsilon1.size)

        para1 = [gamma1, gamma2, epsilon1, epsilon2 + perturb, Groom ,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_epsilon2 = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_epsilon2 = np.reshape(dQdpara_epsilon2, dQdpara_epsilon2.size)

        para1 = [gamma1, gamma2, epsilon1, epsilon2, Groom + perturb,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_Groom = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_Groom = np.reshape(dQdpara_Groom, dQdpara_Groom.size)

        para1 = [gamma1, gamma2, epsilon1, epsilon2, Groom,
                 travelCost + perturb, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_travelCost = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_travelCost = np.reshape(dQdpara_travelCost, dQdpara_travelCost.size)

        para1 = [gamma1, gamma2, epsilon1, epsilon2, Groom,
                 travelCost, pushButtonCost + perturb]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        dQdpara_pushButtonCost = (two1.Qsfm - self.Qsfm) / perturb
        dQdpara_pushButtonCost = np.reshape(dQdpara_pushButtonCost, dQdpara_pushButtonCost.size)

        return dQdpara_gamma1, dQdpara_gamma2, dQdpara_epsilon1, dQdpara_epsilon2, \
               dQdpara_Groom, dQdpara_travelCost, dQdpara_pushButtonCost


    def dpdpara(self):
        dpdQ = self.dpdQ()
        dQdpara_gamma1, dQdpara_gamma2, dQdpara_epsilon1, dQdpara_epsilon2, \
        dQdpara_Groom, dQdpara_travelCost, dQdpara_pushButtonCost = self.dQdpara()

        dpdpara_gamma1 = dpdQ.dot(dQdpara_gamma1)
        dpdpara_gamma2 = dpdQ.dot(dQdpara_gamma2)
        dpdpara_epsilon1 = dpdQ.dot(dQdpara_epsilon1)
        dpdpara_epsilon2 = dpdQ.dot(dQdpara_epsilon2)
        dpdpara_Groom = dpdQ.dot(dQdpara_Groom)
        dpdpara_travelCost = dpdQ.dot(dQdpara_travelCost)
        dpdpara_pushButtonCost = dpdQ.dot(dQdpara_pushButtonCost)

        dpdpara_gamma1 = np.reshape(dpdpara_gamma1, (self.na, self.n))
        dpdpara_gamma2 = np.reshape(dpdpara_gamma2, (self.na, self.n))
        dpdpara_epsilon1 = np.reshape(dpdpara_epsilon1, (self.na, self.n))
        dpdpara_epsilon2 = np.reshape(dpdpara_epsilon2, (self.na, self.n))
        dpdpara_Groom = np.reshape(dpdpara_Groom, (self.na, self.n))
        dpdpara_travelCost = np.reshape(dpdpara_travelCost, (self.na, self.n))
        dpdpara_pushButtonCost = np.reshape(dpdpara_pushButtonCost, (self.na, self.n))

        return dpdpara_gamma1, dpdpara_gamma2, dpdpara_epsilon1, dpdpara_epsilon2, \
               dpdpara_Groom, dpdpara_travelCost, dpdpara_pushButtonCost

    '''
    def dpdQ(self):
    def dQdpara(self):
    def dQdpara_sim(self):
    def dpdpara(self):

    def dQauxdpara(self, obs, para_new):
    def dQauxdpara_sim(self, obs, para_new):
    '''

    def dQauxdpara(self, obs, para_new, initial_valueNew = 0):
        pi = np.ones(self.nq * self.nq) / self.nq / self.nq
        twoboxHMM = HMMtwobox(self.ThA, self.softpolicy, pi)

        twobox_newde = twoboxMDPder(self.discount, self.nq, self.nr,
                                    self.na, self.nl, para_new, initial_valueV = initial_valueNew)

        dQauxdpara_gamma1 = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                    twobox_newde.transitionDerivative()[0], twobox_newde.dpdpara()[0])
        dQauxdpara_gamma2 = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                    twobox_newde.transitionDerivative()[1], twobox_newde.dpdpara()[1])
        dQauxdpara_epsilon1 = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                    twobox_newde.transitionDerivative()[2], twobox_newde.dpdpara()[2])
        dQauxdpara_epsilon2 = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                      twobox_newde.transitionDerivative()[3], twobox_newde.dpdpara()[3])
        dQauxdpara_Groom = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                   np.zeros(twobox_newde.ThA.shape), twobox_newde.dpdpara()[4])
        dQauxdpara_travelCost = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                        np.zeros(twobox_newde.ThA.shape), twobox_newde.dpdpara()[5])
        dQauxdpara_pushButtonCost = twoboxHMM.computeQauxDE(obs, twobox_newde.ThA, twobox_newde.softpolicy,
                                                            np.zeros(twobox_newde.ThA.shape),twobox_newde.dpdpara()[6])

        return dQauxdpara_gamma1, dQauxdpara_gamma2, dQauxdpara_epsilon1, dQauxdpara_epsilon2, \
               dQauxdpara_Groom, dQauxdpara_travelCost, dQauxdpara_pushButtonCost, twobox_newde.Vsfm
        # Also return the value based on the new MDP model, this is to speed up the calculation of the next iteration



    def dQauxdpara_sim(self, obs, para_new):
        pi  = np.ones(self.nq * self.nq) / self.nq / self.nq
        twoboxHMM = HMMtwobox(self.ThA, self.softpolicy, pi)

        perturb = 10 ** -6

        # beta = para_new[0]     # available food dropped back into box after button press
        # gamma1 = para_new[1]   # reward becomes available in box 1
        # gamma2 = para_new[2]   # reward becomes available in box 2
        # delta = para_new[3]    # animal trips, doesn't go to target location
        # direct = para_new[4]   # animal goes right to target, skipping location 0
        # epsilon1 = para_new[5] # available food disappears from box 1
        # epsilon2 = para_new[6] # available food disappears from box 2
        # rho = para_new[7]      # food in mouth is consumed
        #
        # Reward = para_new[8]   # reward per time step with food in mouth
        # Groom = para_new[9]     # location 0 reward
        # travelCost = para_new[10]
        # pushButtonCost = para_new[11]

        beta = 0     # available food dropped back into box after button press
        gamma1 = para_new[0]   # reward becomes available in box 1
        gamma2 = para_new[1]   # reward becomes available in box 2
        delta = 0    # animal trips, doesn't go to target location
        direct = 0   # animal goes right to target, skipping location 0
        epsilon1 = para_new[2] # available food disappears from box 1
        epsilon2 = para_new[3] # available food disappears from box 2
        rho = 1      # food in mouth is consumed
        # State rewards
        Reward = 1   # reward per time step with food in mouth
        Groom = para_new[4]     # location 0 reward
        # Action costs
        travelCost = para_new[5]
        pushButtonCost = para_new[6]

        twobox_new = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para_new)
        twobox_new.setupMDP()
        twobox_new.solveMDP_sfm()
        Qaux = twoboxHMM.computeQaux(obs, twobox_new.ThA, twobox_new.softpolicy)

        para1 = [ gamma1 + perturb, gamma2, epsilon1, epsilon2, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_gamma1 = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2 + perturb, epsilon1, epsilon2, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_gamma2 = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2 , epsilon1+ perturb, epsilon2, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_epsilon1 = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2 , epsilon1, epsilon2+ perturb, Groom,
                 travelCost, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_epsilon2 = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2, epsilon1, epsilon2, Groom + perturb,
                 travelCost , pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_Groom = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2 , epsilon1, epsilon2, Groom,
                 travelCost+ perturb, pushButtonCost]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_travelCost = (Qaux1 - Qaux) / perturb

        para1 = [gamma1, gamma2 , epsilon1, epsilon2, Groom,
                 travelCost, pushButtonCost+ perturb]
        two1 = twoboxMDP(self.discount, self.nq, self.nr, self.na, self.nl, para1)
        two1.setupMDP()
        two1.solveMDP_sfm()
        Qaux1 = twoboxHMM.computeQaux(obs, two1.ThA, two1.softpolicy)
        dQauxdpara_pushButtonCost = (Qaux1 - Qaux) / perturb

        return dQauxdpara_gamma1, dQauxdpara_gamma2, dQauxdpara_epsilon1, dQauxdpara_epsilon2, \
               dQauxdpara_Groom, dQauxdpara_travelCost, dQauxdpara_pushButtonCost