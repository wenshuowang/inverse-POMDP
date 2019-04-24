import numpy as np
from boxtask_func import *

class HMMtwobox:
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi
        self.S = len(self.pi)  # number of possible values of the hidden state
        self.R = 2
        self.L = 3
        self.Ss = int(sqrt(self.S))

    def _states(self, r, l):
        temp = np.reshape(np.array(range(self.Ss)), [1, self.Ss])
        return np.squeeze(l * self.S * self.R + tensorsum(temp * self.R * self.Ss, r * self.Ss + temp)).astype(int)


    def forward(self, obs):
        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]   # location, three possible values

        alpha = np.zeros((self.S, T))  # initialize alpha value for each belief value
        alpha[:, 0] = self.pi * self.B[act[0], self._states(rew[0], loc[0])]

        for t in range(1, T):
            alpha[:,  t] = np.dot(alpha[:, t - 1], self.A[act[t - 1]][
                np.ix_(self._states(rew[t-1], loc[t-1]), self._states(rew[t], loc[t]))]) \
                           * self.B[act[t], self._states(rew[t], loc[t])]
        return alpha


    def backward(self, obs):
        """
        Backward path
        :param obs: a sequence of observations
        :return: predict future observations
        """
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        beta = np.zeros((self.S, T))
        beta[:, -1] = 1
        for t in reversed(range(T - 1)):
            beta[:, t] = np.dot(self.A[act[t]][np.ix_(self._states(rew[t], loc[t]),
                                                      self._states(rew[t+1], loc[t+1]))],
                                beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1], loc[t+1])])

        return beta


    def forward_scale(self, obs):

        T = obs.shape[0]        # length of a sample sequence

        act = obs[:, 0]   # action, two possible values: 0: doing nothing; 1: press button
        rew = obs[:, 1]   # observable, two possible values: 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        alpha = np.zeros((self.S, T))   # initialize alpha value for each belief value
        scale = np.zeros(T)

        alpha[:, 0] = self.pi * self.B[act[0], self._states(rew[0], loc[0])]
        scale[0] = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / scale[0]

        for t in range(1, T):
            alpha[:,  t] = np.dot(alpha[:, t - 1], self.A[act[t - 1]][
                np.ix_(self._states(rew[t-1], loc[t-1]), self._states(rew[t], loc[t]))]) \
                           * self.B[act[t], self._states(rew[t], loc[t])]
            scale[t] = np.sum(alpha[:, t])
            if scale[t] == 0:
                print(t)
            alpha[:, t] = alpha[:, t] / scale[t]

        return alpha, scale


    def backward_scale(self, obs, scale):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        beta = np.zeros((self.S, T))
        beta[:, T - 1] = 1
        #beta[:, T - 1] = beta[:, T - 1] / scale[T - 1]

        for t in reversed(range(T - 1)):
            beta[:, t] = np.dot(self.A[act[t]][np.ix_(self._states(rew[t], loc[t]), self._states(rew[t+1], loc[t+1]))],
                                beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1], loc[t+1])])
            beta[:, t] = beta[:, t] / scale[t + 1]

        return beta


    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, 0)

        return gamma


    def compute_xi(self, alpha, beta, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]   # 0: doing nothing; 1: press button
        rew = obs[:, 1]   # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        xi = np.zeros((T - 1, self.S, self.S))

        for t in range(T - 1):
            xi[t, :, :] = np.diag(alpha[:, t]).dot(
                self.A[act[t]][np.ix_(self._states(rew[t], loc[t]),self._states(rew[t + 1], loc[t+1]))]
            ).dot(np.diag(beta[:, t+1] * self.B[act[t+1], self._states(rew[t+1], loc[t+1])]))
            xi[t, :, :] = xi[t, :, :]/np.sum(xi[t, :, :])

        return xi


    def latent_entr(self, obs):
        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # Entropy of all path that leads to a certain state at t certain time
        Hpath = np.zeros((self.S, T))
        # P(state at time t-1 | state at time t, observations up to time t)
        lat_cond = np.zeros((T - 1, self.S, self.S))

        alpha_scaled, _ = self.forward_scale(obs)
        Hpath[:, 0] = 0

        for t in range(1, T):
            lat_cond[t - 1] = np.diag(alpha_scaled[:, t - 1]).dot(
                self.A[act[t - 1]][np.ix_(self._states(rew[t - 1], loc[t-1]), self._states(rew[t], loc[t]))])
            lat_cond[t - 1] = lat_cond[t - 1] / (
            np.sum(lat_cond[t - 1], axis=0) + 1 * (np.sum(lat_cond[t - 1], axis=0) == 0))

            Hpath[:, t] = Hpath[:, t - 1].dot(lat_cond[t - 1]) - np.sum(
                lat_cond[t - 1] * np.log(lat_cond[t - 1] + 10 ** -13 * (lat_cond[t - 1] == 0)), axis=0)

        lat_ent = np.sum(Hpath[:, -1] * alpha_scaled[:, -1]) - np.sum(
            alpha_scaled[:, -1] * np.log(alpha_scaled[:, -1] + 10 ** -13 * (alpha_scaled[:, -1] == 0)))

        return lat_ent


    def computeQaux(self, obs, Anew, Bnew):
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
        loc = obs[:, 2]  # location, three possible values

        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        Qaux1 = np.sum(np.log(self.pi) * gamma[:, 0])
        Qaux2 = 0
        Qaux3 = 0

        for t in range(T - 1):
            Qaux2 += np.sum(np.log(Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))] +
                                   10 ** -13 * (
                                   Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1], loc[t+1]))] == 0))
                            * xi[t, :, :])

        for t in range(T):
            Qaux3 += np.sum(np.log(Bnew[act[t], self._states(rew[t], loc[t])] +
                                   10 ** -13 * (Bnew[act[t], self._states(rew[t], loc[t])] == 0)) * gamma[:, t])

        Qaux = 1 * (Qaux1 + Qaux2) + 1 * Qaux3

        return Qaux


    def computeQauxDE(self, obs, Anew, Bnew, Anewde, Bnewde):

        T = obs.shape[0]  # length of a sample sequence

        act = obs[:, 0]  # 0: doing nothing; 1: press button
        rew = obs[:, 1]  # 0 : not have; 1: have
        loc = obs[:, 2]  # location, three possible values

        # alpha = self.forward(obs)
        # beta = self.backward(obs)
        alpha, scale = self.forward_scale(obs)
        beta = self.backward_scale(obs, scale)

        gamma = self.compute_gamma(alpha, beta)
        xi = self.compute_xi(alpha, beta, obs)
        dQaux1 = np.sum(np.log(self.pi) * gamma[:, 0])
        dQaux2 = 0
        dQaux3 = 0

        for t in range(T - 1):
            # Qaux2 += np.sum(np.log(10 ** -13 + Anew[act[t]][
            #   np.ix_(self._states(rew[t]), self._states(rew[t + 1]))]) * xi[t, :, :])

            Aelement = Anew[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))]
            Aelement_prime = Aelement + 1 * (Aelement == 0)
            dQaux2_ins = Anewde[act[t]][np.ix_(self._states(rew[t],loc[t]), self._states(rew[t + 1],loc[t+1]))
                         ] / (Aelement_prime) * (Aelement != 0) * xi[t, :, :]
            dQaux2 += np.sum(dQaux2_ins)


            Belement = Bnew[act[t], self._states(rew[t], loc[t])]
            Belement_prime = Belement + 1 * (Belement == 0)
            dQaux3_ins = Bnewde[act[t], self._states(rew[t], loc[t])] / Belement_prime * \
                         (Belement != 0) * gamma[:, t]
            dQaux3 += np.sum(dQaux3_ins)

            dQaux = dQaux1 + dQaux2 + dQaux3

        return dQaux