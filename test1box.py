from __future__ import division
import numpy as np
from scipy.stats import binom
from scipy.integrate import quad
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz, expm
from scipy import optimize
from math import sqrt
from scipy.stats import norm


from boxtask_func import *
from MDPclass import *
from oneboxCol import *
from HMMoneboxCol import *
import timeit


nq = 5
nr = 2
na = 2
discount = 0.99

beta = 0   # available food dropped back into box after button press
gamma = 0.3    # reward becomes available
epsilon = 0.1   # available food disappears
rho = 1    # food in mouth is consumed
pushButtonCost = 0.6
Reward = 1
NumCol = 4
qmin = 0.3
qmax = 0.7
Ncol = NumCol - 1

parameters = [beta, gamma, epsilon, rho, pushButtonCost,
              NumCol, qmin, qmax]

T = 1000
N = 2
hiddenInitial = 0
obsInitial = 0

oneboxColdata = oneboxColMDPdata(discount, nq, nr, na, parameters, T, N)
oneboxColdata.dataGenerate_sfm(hiddenInitial, obsInitial)   #softmax policy

belief = oneboxColdata.belief
action = oneboxColdata.action
reward = oneboxColdata.reward
hybrid = oneboxColdata.hybrid
trueState = oneboxColdata.trueState
color = oneboxColdata.color

obsN = np.dstack([action, reward, color])
latN = belief.copy()


obs = obsN[0]
lat = latN[0]

beta_ini = 0
gamma_ini = 0.2
epsilon_ini = 0.15
rho_ini = 1
pushButtonCost_ini = 0.3
NumCol_ini = NumCol
qmin_ini = 0.2
qmax_ini = 0.6

parameters_ini = [beta_ini, gamma_ini, epsilon_ini, rho_ini, pushButtonCost_ini,
                  NumCol_ini, qmin_ini, qmax_ini]
parameters_old = np.copy(parameters_ini)

print("The true paramters are:               ", parameters )
print("The initial estimation parameters are:", parameters_ini)
print ("Now starting with some initialized value of the parameters, we are going to use EM(G) " \
"algorithm for parameter estimation:")


start = timeit.default_timer()

NN1 = 100  # iteration number for the EM algorithm
eps = 10 ** -6

para_old_traj = []
para_new_traj = []

log_likelihoods_old = []
log_likelihoods_new = []
log_likelihoods_com_old = []  # old posterior, old parameters
log_likelihoods_com_new = []  # old posterior, new parameters
latent_entropies = []

count_E = 0
while count_E < NN1:

    print('\nThe', count_E, '-th iteration of the EM(G) algorithm')

    if count_E == 0:
        parameters_old = np.array(parameters_ini)
    else:
        parameters_old = np.copy(parameters_new)  # update parameters

        if np.max(np.absolute(parameters_old - parameters_new)) < 0.001:
            break

    para_old_traj.append(parameters_old)
    print(parameters_old)

    ##########  E-step ##########

    ## Use old parameters to estimate posterior
    oneboxColder = oneboxColMDPder(discount, nq, nr, na, parameters_old)
    ThA_old = oneboxColder.ThA
    softpolicy_old = oneboxColder.softpolicy
    TBo_old = oneboxColder.Trans_hybrid_obs
    OE_TS_old = oneboxColder.Obs_emis_trans
    pi = np.ones(nq) / nq
    Ncol_old = parameters_old[5].astype(int) - 1
    oneHMMCol = HMMoneboxCol(ThA_old, softpolicy_old, TBo_old, OE_TS_old, pi, Ncol_old)

    ## Calculate likelihood of observed and complete date, and entropy of the latent sequence
    complete_likelihood_old = oneHMMCol.computeQaux(obs, ThA_old, softpolicy_old, TBo_old)
    latent_entropy = oneHMMCol.latent_entr(obs)
    log_likelihood = complete_likelihood_old + latent_entropy

    log_likelihoods_com_old.append(complete_likelihood_old)

    latent_entropies.append(latent_entropy)
    log_likelihoods_old.append(log_likelihood)

    print(complete_likelihood_old)
    print(log_likelihood)

    ## Check convergence
    if len(log_likelihoods_old) >= 2 and np.abs(log_likelihood - log_likelihoods_old[-2]) < eps:
        print('EM has converged!')
        break

    ##########  M(G)-step ##########

    count_M = 0
    para_new_traj.append([])
    log_likelihoods_com_new.append([])
    log_likelihoods_new.append([])
    stepsize = 2 * 10 ** -5

    # Start the gradient descent from the old parameters
    parameters_new = np.copy(parameters_old)
    complete_likelihood_new = complete_likelihood_old
    likelihood = complete_likelihood_new + latent_entropy

    para_new_traj[count_E].append(parameters_new)
    log_likelihoods_com_new[count_E].append(complete_likelihood_new)
    log_likelihoods_new[count_E].append(likelihood)

    print('\n M-step ')
    print(parameters_new)
    print(complete_likelihood_new)
    print(likelihood)

    while True:
        #print(np.array(oneboxColder.dQauxdpara_sim(obs, parameters_new)))
        ## Go the potential next point with gradient descent
        para_temp = parameters_new + stepsize * np.array(oneboxColder.dQauxdpara_sim(obs, parameters_new))
        #temp = np.copy(para_temp)
        #para_temp = np.copy(parameters_old)

        ## Check the ECDLL (old posterior, new parameters)
        onebox_new = oneboxColMDP(discount, nq, nr, na, para_temp)
        onebox_new.setupMDP()
        onebox_new.solveMDP_sfm()
        ThA_new = onebox_new.ThA
        softpolicy_new = onebox_new.softpolicy

        TBo_new = onebox_new.Trans_hybrid_obs
        OE_TS_new = onebox_new.Obs_emis_trans
        pi = np.ones(nq) / nq
        Ncol_new = para_temp[5] - 1
        oneHMMCol_new = HMMoneboxCol(ThA_new, softpolicy_new, TBo_old, OE_TS_new, pi, Ncol_new)

        complete_likelihood_new_temp = oneHMMCol.computeQaux(obs, ThA_new, softpolicy_new, TBo_new)

        ## Update the parameter if the ECDLL can be improved
        if complete_likelihood_new_temp > complete_likelihood_new:
            parameters_new = np.copy(para_temp)

            complete_likelihood_new = complete_likelihood_new_temp
            likelihood = complete_likelihood_new + latent_entropy

            para_new_traj[count_E].append(parameters_new)
            log_likelihoods_com_new[count_E].append(complete_likelihood_new)
            log_likelihoods_new[count_E].append(likelihood)

            print(parameters_new)
            print(complete_likelihood_new)
            print(likelihood)

            count_M += 1
        else:
            stepsize /= 2
            if stepsize < 5 * 10 ** -5:
                break

    count_E += 1

stop = timeit.default_timer()