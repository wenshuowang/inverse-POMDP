from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
from twobox import *
from HMMtwobox import *
import pickle
from datetime import datetime
import os
from pprint import pprint
path = os.getcwd()
datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')


###########################################################
#
#               Pre-process data
#
###########################################################
data = loadmat('NeuralDatafromNeda/behavior74.mat')

location = np.copy(data['bLocX74'][0])

# fill nan with the average of the previous and the next
nanind = np.where(np.isnan(location))[0]
location[nanind] = (location[nanind - 1] + location[nanind + 1]) / 2

# there might be two adjacent nans
nanind1 = np.where(np.isnan(location))[0][::2]
nanind2 = np.where(np.isnan(location))[0][1::2]
location[nanind1] = (location[nanind1 - 1] + location[nanind1 + 2]) / 2
location[nanind2] = location[nanind1]

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# smooothing the location
location = movingaverage(location, 5)
# adjust the value of the outliers
location[np.where(location < 350)[0]] = 1
location[np.where(location > 450)[0]] = 2
location[np.all([location >= 350, location <= 450], axis=0)] = 0

g0 = 1    # g0 = go to location 0
g1 = 2    # g1 = go toward box 1 (via location 0 if from 2)
g2 = 3    # g2 = go toward box 2 (via location 0 if from 1)
loc_change_ind = np.where((location[0:-1] - location[1:])!= 0)[0]
goaction = np.zeros(len(location))
for i in range(len(loc_change_ind)):
    if location[loc_change_ind[i]] == 0:  # at center location
        if location[loc_change_ind[i] + 1] == 1:  # go towards box 1
            goaction[loc_change_ind[i]] = g1
        else:  # go towards box 2
            goaction[loc_change_ind[i]] = g2
    elif location[loc_change_ind[i]] == 1:  # at box1
        if location[loc_change_ind[i] + 1] == 0:  # go towards box 0
            goaction[loc_change_ind[i]] = g0
        else:  # go towards box 1
            goaction[loc_change_ind[i]] = g2
    else:  # at box2
        if location[loc_change_ind[i] + 1] == 0:  # go towards box 0
            goaction[loc_change_ind[i]] = g0
        else:  # go towards box 1
            goaction[loc_change_ind[i]] = g1


b1Pushed = data['b1PushedTimes74'][0] // 200
b2Pushed = data['b2PushedTimes74'][0] // 200
rew1Del = data['rew1DelTimes74'][0] // 200
rew2Del = data['rew2DelTimes74'][0] // 200

pb = 4
action = np.copy(goaction)
for i in range(len(b1Pushed)):
    action[b1Pushed[i]] = pb

for i in range(len(b2Pushed)):
    action[b2Pushed[i]] = pb

rewardDel = np.zeros(len(location))
for i in range(len(rew1Del)):
    rewardDel[rew1Del[i]] = 1

for i in range(len(rew2Del)):
    rewardDel[rew2Del[i]] = 1

T = 15000
loc = location[0:T].astype(int)
act = action[0:T].astype(int)
rew = rewardDel[0:T].astype(int)


#######################
# add=hoc modification of the data
#######################
rew[1224] = 0
rew[1225] = 1
rew[2445] = 0
rew[2446] = 1
rew[3852] = 0
rew[3853] = 1

rew[3384] = 0

rew[11538] = 1
rew[11539] = 0

rew[12620] = 0

rew[13729] = 1
rew[13730] = 0
rew[13731] = 0
#######################



###########################################################
#
#               EM algorithm
#
###########################################################

obsN = np.dstack([act, rew, loc])
obs = obsN[0]

E_MAX_ITER = 500       # 100      # maximum number of iterations of E-step
GD_THRESHOLD = 10 #0.5   # 0.01      # stopping criteria of M-step (gradient descent)
E_EPS = 0.5                         # stopping criteria of E-step
#M_LR_INI = float(sys.argv[1])
M_LR_INI =  1 * 10 ** -6         # initial learning rate in the gradient descent step
LR_DEC =  4                       # number of times that the learning rate can be reduced
SaveEvery = 50
# No need to manual interaction to specify parameters in the command line
#parameters = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost]
parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
                      'GD_THRESHOLD': GD_THRESHOLD,
                      'E_EPS': E_EPS,
                      'M_LR_INI': M_LR_INI,
                      'LR_DEC': LR_DEC,
                      'SaveEvery': SaveEvery,
                      'ParaInitial': [np.array([0.2, 0.25, 0, 0, 0.2, 0.3, 0.5])]
                      # 'ParaInitial': [np.array(list(map(float, i.strip('[]').split(',')))) for i in sys.argv[3].strip('()').split('-')]
                      # Initial parameter is a set that contains arrays of parameters, here only consider one initial point
                      }
output1 = open(path  + '/' + datestring + '_real_ParameterMain_twobox' + '.pkl', 'wb')
pickle.dump(parameterMain_dict, output1)
output1.close()


### Choose which sample is used for inference
sampleIndex = [0]
NN = len(sampleIndex)

### Set initial parameter point
parameters_iniSet = parameterMain_dict['ParaInitial']

discount = 0.99
nq = 5
nr = 2
nl = 3
na = 5

print("\nThe initial points for estimation are:", parameters_iniSet)

#### EM algorithm for parameter estimation
print("\nEM algorithm begins ...")
# NN denotes multiple data set, and MM denotes multiple initial points
# NN_MM_para_old_traj = []
# NN_MM_para_new_traj = []
# NN_MM_log_likelihoods_old = []
# NN_MM_log_likelihoods_new = []
# NN_MM_log_likelihoods_com_old = []    # old posterior, old parameters
# NN_MM_log_likelihoods_com_new = []    # old posterior, new parameters
# NN_MM_latent_entropies = []


for nn in range(NN):

    print("\nFor the", sampleIndex[nn] + 1, "-th set of data:")

    ##############################################################
    # Compute likelihood
    obs = obsN[sampleIndex[nn], :, :]

    MM = len(parameters_iniSet)

    # MM_para_old_traj = []
    # MM_para_new_traj = []
    # MM_log_likelihoods_old = []
    # MM_log_likelihoods_new = []
    # MM_log_likelihoods_com_old = []    # old posterior, old parameters
    # MM_log_likelihoods_com_new = []    # old posterior, new parameters
    # MM_latent_entropies = []

    for mm in range(MM):
        parameters_old = np.copy(parameters_iniSet[mm])

        print("\n######################################################\n",
              mm + 1, "-th initial estimation:", parameters_old)

        itermax = E_MAX_ITER #100  # iteration number for the EM algorithm
        eps = E_EPS   # Stopping criteria for E-step in EM algorithm

        para_old_traj = []
        para_new_traj = []

        log_likelihoods_old = []
        log_likelihoods_new = []
        log_likelihoods_com_old = []  # old posterior, old parameters
        log_likelihoods_com_new = []  # old posterior, new parameters
        latent_entropies = []

        count_E = 0
        while True:

            print("The", count_E + 1, "-th iteration of the EM(G) algorithm")

            if count_E == 0:
                parameters_old = np.copy(parameters_iniSet[mm])
            else:
                parameters_old = np.copy(parameters_new)  # update parameters

            para_old_traj.append(parameters_old)

            ##########  E-step ##########

            ## Use old parameters to estimate posterior
            #twoboxGra = twoboxMDPder(discount, nq, nr, na, nl, parameters_old, vinitial)
            twoboxGra = twoboxMDPder(discount, nq, nr, na, nl, parameters_old)
            ThA_old = twoboxGra.ThA
            softpolicy_old = twoboxGra.softpolicy
            pi = np.ones(nq * nq) / nq / nq
            twoHMM = HMMtwobox(ThA_old, softpolicy_old, pi)

            ## Calculate likelihood of observed and complete date, and entropy of the latent sequence
            complete_likelihood_old = twoHMM.computeQaux(obs, ThA_old, softpolicy_old)
            latent_entropy = twoHMM.latent_entr(obs)
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
            M_thresh = GD_THRESHOLD
            count_M = 0
            vinitial = 0
            para_new_traj.append([])
            log_likelihoods_com_new.append([])
            log_likelihoods_new.append([])

            learnrate_ini = M_LR_INI * np.exp(- count_E // 20)
            learnrate = learnrate_ini

            # Start the gradient descent from the old parameters
            parameters_new = np.copy(parameters_old)
            complete_likelihood_new = complete_likelihood_old
            log_likelihood = complete_likelihood_new + latent_entropy

            para_new_traj[count_E].append(parameters_new)
            log_likelihoods_com_new[count_E].append(complete_likelihood_new)
            log_likelihoods_new[count_E].append(log_likelihood)

            print("\nM-step")
            print(parameters_new)
            print(complete_likelihood_new)
            print(log_likelihood)

            while True:

                # derivative_value = twoboxGra.dQauxdpara(obs, parameters_new, vinitial)
                # # vinitial is value from previous iteration, this is for computational efficiency
                # para_temp = parameters_new + learnrate * np.array(derivative_value[:-1])
                # vinitial = derivative_value[-1]  # value iteration starts with value from previous iteration
                derivative_value = twoboxGra.dQauxdpara_sim(obs, parameters_new)
                # vinitial is value from previous iteration, this is for computational efficiency
                para_temp = parameters_new + learnrate * np.array(derivative_value)

                ## Check the ECDLL (old posterior, new parameters)
                twobox_new = twoboxMDP(discount, nq, nr, na, nl, para_temp)
                twobox_new.setupMDP()
                twobox_new.solveMDP_sfm()
                ThA_new = twobox_new.ThA
                softpolicy_new = twobox_new.softpolicy
                complete_likelihood_new_temp = twoHMM.computeQaux(obs, ThA_new, softpolicy_new)

                print("         ", para_temp)
                print("         ", complete_likelihood_new_temp)

                ## Update the parameter if the ECDLL can be improved
                if complete_likelihood_new_temp > complete_likelihood_new + M_thresh:
                    parameters_new = np.copy(para_temp)
                    complete_likelihood_new = complete_likelihood_new_temp
                    log_likelihood = complete_likelihood_new + latent_entropy

                    para_new_traj[count_E].append(parameters_new)
                    log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                    log_likelihoods_new[count_E].append(log_likelihood)

                    print('\n', parameters_new)
                    print(complete_likelihood_new)
                    print(log_likelihood)

                    count_M += 1
                else:
                    learnrate /= 2
                    if learnrate < learnrate_ini / (2 ** LR_DEC):
                        break

            # every 50 iterations, download data
            if (count_E + 1) % SaveEvery == 0:
                Experiment_dict = {'ParameterTrajectory_Estep': para_old_traj,
                                   'ParameterTrajectory_Mstep': para_new_traj,
                                   'LogLikelihood_Estep': log_likelihoods_old,
                                   'LogLikelihood_Mstep': log_likelihoods_new,
                                   'Complete_LogLikelihood_Estep': log_likelihoods_com_old,
                                   'Complete_LogLikelihood_Mstep': log_likelihoods_com_new,
                                   'Latent_entropies': latent_entropies
                                   }
                output = open(path + '/' + datestring + '_' + str(count_E + 1) + '_real_EM_twobox' + '.pkl', 'wb')
                pickle.dump(Experiment_dict, output)
                output.close()

            count_E += 1


        # save the remainings
        Experiment_dict = {'ParameterTrajectory_Estep': para_old_traj,
                           'ParameterTrajectory_Mstep': para_new_traj,
                           'LogLikelihood_Estep': log_likelihoods_old,
                           'LogLikelihood_Mstep': log_likelihoods_new,
                           'Complete_LogLikelihood_Estep': log_likelihoods_com_old,
                           'Complete_LogLikelihood_Mstep': log_likelihoods_com_new,
                           'Latent_entropies': latent_entropies
                           }
        output = open(path + '/' + datestring + '_' + str(count_E + 1) + '_real_EM_twobox' + '.pkl', 'wb')
        pickle.dump(Experiment_dict, output)
        output.close()

    #     MM_para_old_traj.append(para_old_traj)  # parameter trajectories for a particular set of data
    #     MM_para_new_traj.append(para_new_traj)
    #     MM_log_likelihoods_old.append(log_likelihoods_old)  # likelihood trajectories for a particular set of data
    #     MM_log_likelihoods_new.append(log_likelihoods_new)
    #     MM_log_likelihoods_com_old.append(log_likelihoods_com_old)    # old posterior, old parameters
    #     MM_log_likelihoods_com_new.append(log_likelihoods_com_new)    # old posterior, new parameters
    #     MM_latent_entropies.append(latent_entropies)
    #
    # NN_MM_para_old_traj.append(MM_para_old_traj)  # parameter trajectories for all data
    # NN_MM_para_new_traj.append(MM_para_new_traj)
    # NN_MM_log_likelihoods_old.append(MM_log_likelihoods_old)  # likelihood trajectories for
    # NN_MM_log_likelihoods_new.append(MM_log_likelihoods_new)
    # NN_MM_log_likelihoods_com_old.append(MM_log_likelihoods_com_old)   # old posterior, old parameters
    # NN_MM_log_likelihoods_com_new.append(MM_log_likelihoods_com_new)   # old posterior, new parameters
    # NN_MM_latent_entropies.append(MM_latent_entropies)




###########################################################
#
#               save data
#
###########################################################

# ## save the running data
# Experiment_dict = {'ParameterTrajectory_Estep': NN_MM_para_old_traj,
#                    'ParameterTrajectory_Mstep': NN_MM_para_new_traj,
#                    'LogLikelihood_Estep': NN_MM_log_likelihoods_old,
#                    'LogLikelihood_Mstep': NN_MM_log_likelihoods_new,
#                    'Complete_LogLikelihood_Estep': NN_MM_log_likelihoods_com_old,
#                    'Complete_LogLikelihood_Mstep': NN_MM_log_likelihoods_com_new,
#                    'Latent_entropies': NN_MM_latent_entropies
#                    }
# output = open(path  + '/' + datestring + '_real_EM_twobox' + '.pkl', 'wb')
# pickle.dump(Experiment_dict, output)
# output.close()
#
# ## save running parameters
# # parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
# #                       'GD_THRESHOLD': GD_THRESHOLD,
# #                       'E_EPS': E_EPS,
# #                       'M_LR_INI': M_LR_INI,
# #                       'LR_DEC': LR_DEC,
# #                       'ParaInitial': parameters_iniSet}
# output1 = open(path  + '/' + datestring + '_real_ParameterMain_twobox' + '.pkl', 'wb')
# pickle.dump(parameterMain_dict, output1)
# output1.close()

print("finish")



# ###########################################################
# #
# #         retrieve data and look into contour
# #
# ###########################################################
# EM_pkl_file = open(path + '/real_EM_twobox.pkl', 'rb')
# EM_pkl = pickle.load(EM_pkl_file)
# EM_pkl_file.close()
#
# ParameterMain_pkl_file = open(path + '/real_ParameterMain_twobox.pkl', 'rb')
# ParameterMain_pkl = pickle.load(ParameterMain_pkl_file)
# ParameterMain_pkl_file.close()
#
# NN_MM_para_old_traj = EM_pkl['ParameterTrajectory_Estep']
# NN_MM_para_new_traj = EM_pkl['ParameterTrajectory_Mstep']
# NN_MM_log_likelihoods_old = EM_pkl['LogLikelihood_Estep']
# NN_MM_log_likelihoods_new = EM_pkl['LogLikelihood_Mstep']
# NN_MM_log_likelihoods_com_old = EM_pkl['Complete_LogLikelihood_Estep']
# NN_MM_log_likelihoods_com_new = EM_pkl['Complete_LogLikelihood_Mstep']
# NN_MM_latent_entropies = EM_pkl['Latent_entropies']
#
# para_traj = [k for i in NN_MM_para_new_traj[0] for j in i  for k in j]
# point = np.copy(para_traj)
#
#
#
# ###################################################################
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# pca = PCA(n_components = 2)
# pca.fit(point - point[-1])
# projectionMat = pca.components_
# print(projectionMat)
#
# # Contour of the likelihood
# step1 = 0.04  # for u (1st principle component)
# step2 = 0.04  # for v (2nd principle component)
# N1 = 25
# N2 = 10
# uOffset =  - step1 * N1 / 2
# vOffset =  - step2 * N2 / 2
#
# uValue = np.zeros(N1)
# vValue = np.zeros(N2)
# Qaux1 = np.zeros((N2, N1))  # Likelihood with ground truth latent
# Qaux2 = np.zeros((N2, N1))  # Expected complete data likelihood
# Qaux3 = np.zeros((N2, N1))  # Entropy of latent posterior
# para_slice = []
#
# for i in range(N1):
#     uValue[i] = step1 * (i) + uOffset
#     for j in range(N2):
#         vValue[j] = step2 * (j) + vOffset
#
#         para_slicePoints = point[-1] + uValue[i] * projectionMat[0] + vValue[j] * projectionMat[1]
#         para_slice.append(para_slicePoints)
#         para = np.copy(para_slicePoints)
#         # print(para)
#
#         twobox = twoboxMDP(discount, nq, nr, na, nl, para)
#         twobox.setupMDP()
#         twobox.solveMDP_sfm()
#         ThA = twobox.ThA
#         policy = twobox.softpolicy
#         pi = np.ones(nq * nq) / nq / nq  # initialize the estimation of the belief state
#         twoboxHMM = HMMtwobox(ThA, policy, pi)
#
#         # Qaux1[j, i] = twoboxHMM.likelihood(lat, obs, ThA, policy)  #given latent state
#         Qaux2[j, i] = twoboxHMM.computeQaux(obs, ThA, policy)
#         Qaux3[j, i] = twoboxHMM.latent_entr(obs)
#
# Loglikelihood = Qaux2 + Qaux3
#
#
# Contour_dict = {'uValue': uValue, 'vValue': vValue, 'Qaux2': Qaux2, 'Qaux3': Qaux3}
# output = open(path  + '/' + datestring + '_real_contour' + '.pkl', 'wb')
# pickle.dump(Contour_dict, output)
# output.close()
#
# # project the trajectories onto the plane
# point_2d = projectionMat.dot((point - point[-1]).T).T
# # true parameters projected onto the plane
# #true_2d = projectionMat.dot(parameters - point[-1])
# fig, ax = plt.subplots(figsize = (10, 10))
# uValuemesh, vValuemesh = np.meshgrid(uValue, vValue)
# cs3 = plt.contour(uValuemesh, vValuemesh, Loglikelihood,
#                   np.arange(np.min(Loglikelihood), np.max(Loglikelihood), 5), cmap='jet')
# #plt.xticks(np.arange(0, 1, 0.1))
# #plt.yticks(np.arange(0, 1, 0.1))
# plt.plot(point_2d[:, 0], point_2d[:, 1], marker='.', color = 'b')   # projected trajectories
# plt.plot(point_2d[-1, 0], point_2d[-1, 1], marker='*', color = 'g', markersize = 10)        # final point
# #plt.plot(true_2d[0], true_2d[1], marker='o', color = 'g')           # true
# ax.grid()
# ax.set_title('Likelihood of observed data')
# plt.xlabel(r'$u \mathbf{\theta}$', fontsize = 10)
# plt.ylabel(r'$v \mathbf{\theta}$', fontsize = 10)
# plt.clabel(cs3, inline=1, fontsize=10)
# plt.colorbar()
# plt.show()
#
# #################################################################
# showlen = 200
# showT = range(1000,1000+showlen)
# para_est = point[-1]
# twobox_est = twoboxMDP(discount, nq, nr, na, nl, para_est)
# twobox_est.setupMDP()
# twobox_est.solveMDP_sfm()
# ThA = twobox_est.ThA
# policy = twobox_est.softpolicy
# pi = np.ones(nq * nq)/ nq /nq  # initialize the estimation of the belief state
# twoboxHMM_est = HMMtwobox(ThA, policy, pi)
#
# alpha_est, scale_est = twoboxHMM_est.forward_scale(obs)
# beta_est = twoboxHMM_est.backward_scale(obs, scale_est)
# gamma_est = twoboxHMM_est.compute_gamma(alpha_est, beta_est)
# xi_est = twoboxHMM_est.compute_xi(alpha_est, beta_est, obs)
#
# #lat_compound = nq * lat[:, 0] + lat[:, 1]
#
# fig, ax = plt.subplots(figsize= (20, 10))
# plt.imshow(gamma_est[:, showT], interpolation='Nearest', cmap='gray')
# #plt.plot(lat_compound[showT], color = 'r',marker ='.', markersize = 15)
# plt.xticks(np.arange(0, showlen, 10))
# plt.xlabel('time')
# plt.ylabel('belief state')
# plt.show()
#
# belief1_est = np.sum(np.reshape(gamma_est[:, showT].T, (showlen, nq, nq)), axis = 2)
# belief2_est = np.sum(np.reshape(gamma_est[:, showT].T, (showlen, nq, nq)), axis = 1)
#
# fig = plt.figure(figsize= (20, 4))
# ax1 = fig.add_subplot(211)
# ax1.imshow(belief1_est.T, interpolation='Nearest', cmap='gray')
# ax1.set(title = 'belief of box 1 based on estimated parameters')
# ax2 = fig.add_subplot(212)
# ax2.imshow(belief2_est.T, interpolation='Nearest', cmap='gray')
# ax2.set(title = 'belief of box 2 based on estimated parameters')
# plt.show()


print('hello')

