from twoboxCol import *
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from HMMtwoboxCol import *


discount = 0.99  # temporal discount , used to solve the MDP with value iteration

nq = 5  # number of belief states per box
nr = 2  # number of reward states
nl = 3  # number of location states
na = 5

beta = 0   # available food dropped back into box after button press
gamma1 = 0.3    # reward becomes available
epsilon1 = 0.1   # available food disappears
gamma2 = 0.3    # reward becomes available
epsilon2 = 0.1   # available food disappears
rho = 1    # food in mouth is consumed
pushButtonCost = 0.6
Reward = 1
groom = 0.05     # location 0 reward
travelCost = 0.2
pushButtonCost = 0.3

NumCol = 4
qmin = 0.3
qmax = 0.7
Ncol = NumCol - 1

parameters = [gamma1, gamma2, epsilon1, epsilon2,
              groom, travelCost, pushButtonCost, NumCol, qmin, qmax]

T = 10
N = 1
twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parameters, T, N)
twoboxColdata.dataGenerate_sfm(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)

hybrid = twoboxColdata.hybrid
action = twoboxColdata.action
location = twoboxColdata.location
belief1 = twoboxColdata.belief1
belief2 = twoboxColdata.belief2
reward = twoboxColdata.reward
trueState1 = twoboxColdata.trueState1
trueState2 = twoboxColdata.trueState2
color1 = twoboxColdata.color1
color2 = twoboxColdata.color2

obsN = np.dstack([action, reward, location, color1, color2])  # includes the action and the observable states
latN = np.dstack([belief1, belief2])
truthN = np.dstack([trueState1, trueState2])

obs = obsN[0]
lat = latN[0]


# trueState1 = truthN[0, :, 0]
# trueState2 = truthN[0, :, 1]
#
# action = obsN[0, :, 0]
# reward = obsN[0, :, 1]
# location = obsN[0, :, 2]
# color1 = obsN[0, :, 3]
# color2 = obsN[0, :, 4]
#
# belief1 = latN[0, :, 0]
# belief2 = latN[0, :, 1]

twoboxCol = twoboxColMDPder(discount, nq, nr, na, nl, parameters)
#twoboxCol.setupMDP()
#twoboxCol.solveMDP_sfm(initial_value = 0)
pi = np.ones(nq * nq) / nq / nq
twoColHMM = HMMtwoboxCol(twoboxCol.ThA, twoboxCol.softpolicy, twoboxCol.Trans_hybrid_obs12,
                         twoboxCol.Obs_emis_trans1, twoboxCol.Obs_emis_trans2, pi, NumCol - 1)

###################    EM algorithm     ###########################
gamma1_ini = 0.5
epsilon1_ini = 0.3
gamma2_ini = 0.5
epsilon2_ini = 0.3
groomCost_ini = 0.2
travelCost_ini = 0.1
pushButtonCost_ini = 0.4
NumCol_ini = NumCol
qmin_ini = 0.2
qmax_ini = 0.7

#parameters_ini = [gamma1_ini, gamma2_ini, epsilon1_ini, epsilon2_ini,
#                  groomCost_ini, pushButtonCost_ini, pushButtonCost_ini,
#                  NumCol_ini, qmin_ini, qmax_ini]
parameters_ini = np.copy(parameters)
parameters_old = np.copy(parameters_ini)

print("The true paramters are:               ", parameters )
print("The initial estimation parameters are:", parameters_ini)
print ("Now starting with some initialized value of the parameters, we are going to use EM(G) " \
"algorithm for parameter estimation:")

sampleIndex = [0]
NN = len(sampleIndex)

parameters_iniSet = [parameters_ini]
E_MAX_ITER = 300       # 100    # maximum number of iterations of E-step
GD_THRESHOLD = 0.015   # 0.01      # stopping criteria of M-step (gradient descent)
E_EPS = 10 ** -8                  # stopping criteria of E-step
M_LR_INI = 5 * 10 ** -6           # initial learning rate in the gradient descent step
LR_DEC =  3                       # number of times that the learning rate can be reduced

#### EM algorithm for parameter estimation
print("\nEM algorithm begins ...")
# NN denotes multiple data set, and MM denotes multiple initial points
NN_MM_para_old_traj = []

NN_MM_para_new_traj = []
NN_MM_log_likelihoods_old = []
NN_MM_log_likelihoods_new = []
NN_MM_log_likelihoods_com_old = []  # old posterior, old parameters
NN_MM_log_likelihoods_com_new = []  # old posterior, new parameters
NN_MM_latent_entropies = []

NN_likelihoods = []

for nn in range(NN):

    print("\nFor the", sampleIndex[nn] + 1, "-th set of data:")

    ##############################################################
    # Compute likelihood
    lat = latN[sampleIndex[nn]]
    obs = obsN[sampleIndex[nn], :, :]

    MM = len(parameters_iniSet)

    MM_para_old_traj = []
    MM_para_new_traj = []
    MM_log_likelihoods_old = []
    MM_log_likelihoods_new = []
    MM_log_likelihoods_com_old = []  # old posterior, old parameters
    MM_log_likelihoods_com_new = []  # old posterior, new parameters
    MM_latent_entropies = []

    for mm in range(MM):
        parameters_old = np.copy(parameters_iniSet[mm])

        print("\n", mm + 1, "-th initial estimation:", parameters_old)

        itermax = E_MAX_ITER  # 100  # iteration number for the EM algorithm
        eps = E_EPS  # Stopping criteria for E-step in EM algorithm

        para_old_traj = []
        para_new_traj = []

        log_likelihoods_old = []
        log_likelihoods_new = []
        log_likelihoods_com_old = []  # old posterior, old parameters
        log_likelihoods_com_new = []  # old posterior, new parameters
        latent_entropies = []

        count_E = 0
        while count_E < itermax:

            print("\n The", count_E + 1, "-th iteration of the EM(G) algorithm")

            if count_E == 0:
                parameters_old = np.copy(parameters_iniSet[mm])
            else:
                parameters_old = np.copy(parameters_new)  # update parameters

            para_old_traj.append(parameters_old)

            ##########  E-step ##########

            ## Use old parameters to estimate posterior

            # twoboxGra = twoboxMDPder(discount, nq, nr, na, nl, parameters_old, vinitial)
            twoboxColGra = twoboxColMDPder(discount, nq, nr, na, nl, parameters_old)
            ThA_old = twoboxColGra.ThA
            softpolicy_old = twoboxColGra.softpolicy
            Trans_hybrid_obs12_old = twoboxColGra.Trans_hybrid_obs12
            Obs_emis_trans1_old = twoboxColGra.Obs_emis_trans1
            Obs_emis_trans2_old = twoboxColGra.Obs_emis_trans2
            Ncol_old = parameters_old[7].astype(int) - 1
            pi = np.ones(nq * nq) / nq / nq
            twoColHMM = HMMtwoboxCol(ThA_old, softpolicy_old, Trans_hybrid_obs12_old,
                                     Obs_emis_trans1_old, Obs_emis_trans2_old, pi, Ncol_old)

            ## Calculate likelihood of observed and complete date, and entropy of the latent sequence
            complete_likelihood_old = twoColHMM.computeQaux(obs, ThA_old, softpolicy_old, Trans_hybrid_obs12_old)
            latent_entropy = twoColHMM.latent_entr(obs)
            log_likelihood = complete_likelihood_old + latent_entropy

            log_likelihoods_com_old.append(complete_likelihood_old)
            latent_entropies.append(latent_entropy)
            log_likelihoods_old.append(log_likelihood)

            print(parameters_old)
            print(complete_likelihood_old)
            print(log_likelihood)

            ## Check convergence
            if len(log_likelihoods_old) >= 2 and np.abs(log_likelihood - log_likelihoods_old[-2]) < eps:
                print("EM has converged!")
                break

            ##########  M(G)-step ##########

            count_M = 0
            vinitial = 0
            para_new_traj.append([])
            log_likelihoods_com_new.append([])
            log_likelihoods_new.append([])

            learnrate_ini = M_LR_INI
            learnrate = learnrate_ini

            # Start the gradient descent from the old parameters
            parameters_new = np.copy(parameters_old)
            complete_likelihood_new = complete_likelihood_old
            log_likelihood = complete_likelihood_new + latent_entropy

            para_new_traj[count_E].append(parameters_new)
            log_likelihoods_com_new[count_E].append(complete_likelihood_new)
            log_likelihoods_new[count_E].append(log_likelihood)

            print("    M-step")
            print("     ", parameters_new)
            print("     ", complete_likelihood_new)
            print("     ", log_likelihood)

            while True:

                derivative_value = twoboxColGra.dQauxdpara_sim(obs, parameters_new)
                # vinitial is value from previous iteration, this is for computational efficiency
                para_temp = parameters_new + learnrate * np.array(derivative_value)
                #vinitial = derivative_value[-1]  # value iteration starts with value from previous iteration

                ## Check the ECDLL (old posterior, new parameters)
                twoboxCol_new = twoboxColMDP(discount, nq, nr, na, nl, para_temp)
                twoboxCol_new.setupMDP()
                twoboxCol_new.solveMDP_sfm()
                ThA_new = twoboxCol_new.ThA
                softpolicy_new = twoboxCol_new.softpolicy
                Trans_hybrid_obs12_new = twoboxCol_new.Trans_hybrid_obs12
                complete_likelihood_new_temp = twoColHMM.computeQaux(obs, ThA_new,
                                                                     softpolicy_new, Trans_hybrid_obs12_new)

                print("         ", para_temp)
                print("         ", complete_likelihood_new_temp)

                ## Update the parameter if the ECDLL can be improved
                if complete_likelihood_new_temp > complete_likelihood_new + GD_THRESHOLD:
                    parameters_new = np.copy(para_temp)
                    complete_likelihood_new = complete_likelihood_new_temp
                    log_likelihood = complete_likelihood_new + latent_entropy

                    para_new_traj[count_E].append(parameters_new)
                    log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                    log_likelihoods_new[count_E].append(log_likelihood)

                    print("     ", parameters_new)
                    print("     ", complete_likelihood_new)
                    print("     ", log_likelihood)

                    count_M += 1
                else:
                    learnrate /= 2
                    if learnrate < learnrate_ini / (2 ** LR_DEC):
                        break

            count_E += 1

        MM_para_old_traj.append(para_old_traj)  # parameter trajectories for a particular set of data
        MM_para_new_traj.append(para_new_traj)
        MM_log_likelihoods_old.append(log_likelihoods_old)  # likelihood trajectories for a particular set of data
        MM_log_likelihoods_new.append(log_likelihoods_new)
        MM_log_likelihoods_com_old.append(log_likelihoods_com_old)  # old posterior, old parameters
        MM_log_likelihoods_com_new.append(log_likelihoods_com_new)  # old posterior, new parameters
        MM_latent_entropies.append(latent_entropies)

    NN_MM_para_old_traj.append(MM_para_old_traj)  # parameter trajectories for all data
    NN_MM_para_new_traj.append(MM_para_new_traj)
    NN_MM_log_likelihoods_old.append(MM_log_likelihoods_old)  # likelihood trajectories for
    NN_MM_log_likelihoods_new.append(MM_log_likelihoods_new)
    NN_MM_log_likelihoods_com_old.append(MM_log_likelihoods_com_old)  # old posterior, old parameters
    NN_MM_log_likelihoods_com_new.append(MM_log_likelihoods_com_new)  # old posterior, new parameters
    NN_MM_latent_entropies.append(MM_latent_entropies)



print("end")