import numpy as np


class HMMoneboxCol:
    def __init__(self, A, B, C, D, pi, Ncol):
        self.A = A   #transition probability
        self.B = B   #policy
        self.C = C   # (Trans_hybrid_obs)
        self.D = D   # (Obs_emis.dot(Trans_state, to calculate oberservation emission)
        self.pi = pi
        self.Ncol = Ncol
        self.S = len(self.pi)  # number of possible values of the hidden state

    def _states(self, r):
        return range(self.S * r, self.S * (r+1))

    def forward(self, obs):
        """
        the forward path, used to estimate the state at a given time given all the observations
        with both filtering and smoothing
        :param obs: a sequence of observations
        :return: smoothed probability of state at a certain time
        """

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        col = obs[:, 2]

        alpha = np.zeros((self.S, T))   # initialize alpha value for each belief value
        alpha[:, 0] = self.pi * self.B[act[0], self._states(rew[0])]
        belief_vector = np.array([np.arange(0, 1  ,1/self.S) + 1/self.S/2, 1 - np.arange(0, 1  ,1/self.S) - 1/self.S/2])

        for t in range(1, T):
            if act[t - 1] == 1 and col[t] == self.Ncol:
                alpha[:, t] = np.dot(alpha[:, t - 1], self.A[act[t-1]][
                        np.ix_(self._states(rew[t - 1]), self._states(rew[t]))]) \
                                  * self.B[act[t], self._states(rew[t])]
                #else:
                #    alpha[:, t] = 0
            else:
                alpha[:,  t] = np.dot(alpha[:, t - 1] * (self.D[col[t]].dot(belief_vector)),
                                      self.C[col[t]][np.ix_(self._states(rew[t-1]), self._states(rew[t]))]) \
                           * self.B[act[t], self._states(rew[t])]
        return alpha


    def forward_scale(self, obs):

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        col = obs[:, 2]

        alpha = np.zeros((self.S, T))   # initialize alpha value for each belief value
        scale = np.zeros(T)

        alpha[:, 0] = self.pi * self.B[act[0], self._states(rew[0])]
        scale[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in range(1, T):
            #print(t)
            if act[t - 1] == 1 and col[t] == self.Ncol:
                alpha[:, t] = np.dot(alpha[:, t - 1], self.A[act[t-1]][
                        np.ix_(self._states(rew[t - 1]), self._states(rew[t]))]) \
                                  * self.B[act[t], self._states(rew[t])]
                #else:
                #    alpha[:, t] = 0
            else:
                alpha[:,  t] = np.dot(alpha[:, t - 1] * (self.D[col[t]].dot(belief_vector)),
                                      self.C[col[t]][np.ix_(self._states(rew[t-1]), self._states(rew[t]))]) \
                           * self.B[act[t], self._states(rew[t])]
            scale[t] = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / scale[t]

        return alpha, scale


    def backward(self, obs):
        """
        Backward path
        :param obs: a sequence of observations
        :return: predict future observations
        """
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        col = obs[:, 2]

        beta = np.zeros((self.S, T))
        beta[:, -1] = 1

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in reversed(range(T - 1)):
            if act[t] == 1 and col[t+1] == self.Ncol:
                beta[:, t] = np.dot(self.A[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t+1]))],
                                beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])])
                #else:
                #    beta[:,t] = 0
            else:
                beta[:, t] = np.dot(self.C[col[t+1]][np.ix_(self._states(rew[t]), self._states(rew[t+1]))],
                                    beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])]) * (self.D[col[t + 1]].dot(belief_vector))

        return beta


    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        col = obs[:, 2]

        beta = np.zeros((self.S, T))
        beta[:, T - 1] = 1
        #beta[:, T - 1] = beta[:, T - 1] / scale[T - 1]
        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in reversed(range(T - 1)):
            if act[t] == 1 and col[t+1] == self.Ncol:
                beta[:, t] = np.dot(self.A[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t+1]))],
                                beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])])
                #else:
                #    beta[:,t] = 0
            else:
                beta[:, t] = np.dot(self.C[col[t+1]][np.ix_(self._states(rew[t]), self._states(rew[t+1]))],
                                    beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])]) * (self.D[col[t + 1]].dot(belief_vector))
            beta[:, t] = beta[:, t] / scale[t + 1]

        return beta

    def observation_prob(self, obs):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self.forward(obs)[:, -1])


    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, 0)

        return gamma

    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        col = obs[:, 2]

        xi = np.zeros((T - 1, self.S, self.S))

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in range(T - 1):
            if act[t] == 1 and col[t + 1] == self.Ncol:
                xi[t, :, :] = np.diag(alpha[:, t]).dot(self.A[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]
                                                   ).dot(np.diag(beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])]))
            else:
                xi[t, :, :] = np.diag(alpha[:, t] * (self.D[col[t + 1]].dot(belief_vector))).dot(
                    self.C[col[t+1]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]
                                                   ).dot(np.diag(beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1])]))

            xi[t, :, :] = xi[t, :, :]/np.sum(xi[t, :, :])

        return xi

    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        col = obs[:, 2]

        # Entropy of all path that leads to a certain state at t certain time
        Hpath = np.zeros((self.S, T))
        # P(state at time t-1 | state at time t, observations up to time t)
        lat_cond = np.zeros((T - 1, self.S, self.S))

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath[:, 0] = 0

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in range(1, T):
            if act[t - 1] == 1 and col[t] == self.Ncol:
                lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1]).dot(self.A[act[t - 1]][np.ix_(self._states(rew[t - 1]), self._states(rew[t]))])
            else:
                lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1] * (self.D[col[t]].dot(belief_vector))
                                          ).dot(self.C[col[t]][np.ix_(self._states(rew[t - 1]), self._states(rew[t]))])

            lat_cond[t - 1] = lat_cond[t - 1] / (np.sum(lat_cond[t - 1], axis = 0) + 1 * (np.sum(lat_cond[t - 1], axis = 0) == 0))

            Hpath[:, t] = Hpath[:, t - 1].dot(lat_cond[t - 1]) - np.sum(lat_cond[t - 1] * np.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1]==0)), axis = 0)

        lat_ent = np.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - np.sum(alpha_scaled[:, -1] * np.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

        return lat_ent


    def computeQaux(self, obs, Anew, Bnew, Cnew):
        '''
        computer the Q auxillary funciton, the expected complete data likelihood
        :param obs: observation sequence, used to calculate alpha, beta, gamma, xi
        :param Anew: updated A transition matrix
        :param Bnew: updated B emission matrix
        :return: Q auxilary value
        '''
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        col = obs[:, 2]

        #alpha = self.forward(obs)
        #beta = self.backward(obs)
        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        Qaux1 = np.sum(np.log(self.pi) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0
        Qaux4 = 0

        #xi_delta = np.zeros((T, self.S, self.S))

        for t in range(T - 1):
            #Qaux2 += np.sum(np.log(10 ** -13 + Anew[act[t]][
            #   np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]) * xi[t, :, :])
            if act[t] == 1:
                Trantemp = Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]
            else:
                Trantemp = Cnew[col[t+1]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]

            Qaux2 += np.sum(np.log(Trantemp + 10 ** -13 * (Trantemp == 0)) * xi[t, :, :])

            #xi_delta[t, lat[t], lat[t+1]] = 1
            #Qaux2 += np.sum(np.log(Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] +
            #                       1 * (Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] == 0))
            #                * xi_delta[t])    #to check the code for computing the Qaux

        for t in range(T):
            #Qaux3 += np.sum(np.log(10 ** -13 + Bnew[act[t], self._states(rew[t])]) * gamma[:, t])

             Qaux3 += np.sum(np.log(Bnew[act[t], self._states(rew[t])] +
                                    10 ** -13 * ( Bnew[act[t], self._states(rew[t])] == 0)) * gamma[:, t])

        belief_vector = np.array(
            [np.arange(0, 1, 1 / self.S) + 1 / self.S / 2, 1 - np.arange(0, 1, 1 / self.S) - 1 / self.S / 2])

        for t in range(T - 1):
            if act[t] == 1:
                obstemp = np.ones(self.S)
            else:
                obstemp = self.D[col[t]].dot(belief_vector)

            Qaux4 += np.sum(np.log(obstemp) * gamma[:, t])



        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3 + Qaux4
        #print alpha
        #print beta
        #print Qaux1, Qaux2, Qaux3

        return Qaux


    # def computeQauxDE(self, obs, Anew, Bnew, Anewde, Bnewde):
    #
    #
    #     T = obs.shape[0]  # length of a sample sequence
    #
    #     act = obs[:, 0]  # 0: doing nothing; 1: press button
    #     rew = obs[:, 1]  # 0 : not have; 1: have
    #
    #     #alpha = self.forward(obs)
    #     #beta = self.backward(obs)
    #     alpha, scale = self.forward_scale(obs)
    #     beta = self.backward_scale(obs, scale)
    #
    #     gamma = self.compute_gamma(alpha, beta)
    #     xi = self.compute_xi(alpha, beta, obs)
    #     dQaux1 = 0
    #     dQaux2 = 0
    #     dQaux3 = 0
    #     for t in range(T - 1):
    #         Aelement = Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]
    #         Aelement_prime = Aelement + 1 * (Aelement == 0)
    #         dQaux2_ins = Anewde[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))
    #                      ] / (Aelement_prime) * (Aelement != 0) * xi[t, :, :]
    #
    #         #dQaux2_ins = Anewde[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))
    #         #             ] / (Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]) * \
    #         #             (Anew[act[t]][np.ix_(self._states(rew[t]), self._states(rew[t + 1]))] <> 0 ) \
    #         #             * xi[t, :, :]
    #         dQaux2 += np.sum(dQaux2_ins)
    #
    #     for t in range(T):
    #         Belement = Bnew[act[t], self._states(rew[t])]
    #         Belement_prime = Belement + 1 * (Belement == 0)
    #         dQaux3_ins = Bnewde[act[t], self._states(rew[t])] / Belement * \
    #                      (Belement != 0) * gamma[:, t]
    #
    #         #dQaux3_ins = Bnewde[act[t], self._states(rew[t])] / (Bnew[act[t], self._states(rew[t])]) * \
    #         #              (Bnew[act[t], self._states(rew[t])] <> 0) * gamma[:, t]
    #         dQaux3 += np.sum(dQaux3_ins)
    #
    #     dQaux = dQaux1 + dQaux2 + dQaux3
    #
    #     return dQaux
