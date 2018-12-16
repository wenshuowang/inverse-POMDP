from twobox import *
from HMMtwobox import *
import pickle
import logger
import sys
from datetime import datetime
import os

import timeit
#from twoboxMDPsolver import *

E_MAX_ITER = 300       # 100    # maximum number of iterations of E-step
GD_THRESHOLD = 0.015   # 0.01      # stopping criteria of M-step (gradient descent)
E_EPS = 10 ** -8                  # stopping criteria of E-step
M_LR_INI = 2 * 10 ** -5           # initial learning rate in the gradient descent step
LR_DEC =  4                       # number of times that the learning rate can be reduced

def main():

    ## Save the output log
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')
    #sys.stdout = logger.Logger(datestring)  # output will be both on the screen and in the log file

    parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
                          'GD_THRESHOLD': GD_THRESHOLD,
                          'E_EPS': E_EPS,
                          'M_LR_INI': M_LR_INI,
                          'LR_DEC': LR_DEC,
                          'dataSet': sys.argv[1],
                          'optimizer': sys.argv[2],   # GD: standard gradient descent, PGD: projected GD
                          'sampleIndex': list(map(int, sys.argv[3].strip('[]').split(','))),
                          #'sampleIndex': sys.argv[3],
                          'ParaInitial': [np.array(list(map(float, sys.argv[4].strip('[]').split(','))))]
                          # Initial parameter is a set that contains arrays of parameters, here only consider one initial point
                          }
    ####
    # For data set A, GD, 0, [0.2,0.2,0.15,0.15,0.03,0.2,0.3]
    ####

    ### Choose the data set that will be used
    dataSet = parameterMain_dict['dataSet']
    #print("Using data set", dataSet)

    path = os.getcwd()
    if dataSet == 'A':
        pkl_file = open(path + '/Data/01042018(2215)_dataN_twobox.pkl', 'rb')
        pkl_file1 = open(path + '/Data/01042018(2215)_para_twobox.pkl', 'rb')
    elif dataSet == 'B':
        pkl_file = open(path + '/Data/01062018(2250)_dataN_twobox.pkl', 'rb')
        pkl_file1 = open(path + '/Data/01062018(2250)_para_twobox.pkl', 'rb')

    ### read data from file
    print("Get data from file...")
    dataN_pkl = pickle.load(pkl_file)
    pkl_file.close()

    obsN = dataN_pkl['observations']
    latN = dataN_pkl['beliefs']
    truthN = dataN_pkl['trueStates']
    dataN = dataN_pkl['allData']

    ### Choose which sample is used for inference
    #T = dataN.shape[1]
    sampleIndex = parameterMain_dict['sampleIndex']   # a list of sample index
    NN = len(sampleIndex)
    #NN = 1
    print("Using data", dataSet, sampleIndex)

    #N = len(parameterMain_dict['sampleIndex'])

    #if T > dataN.shape[1]:
    #    sys.exit("The sample length exceeds the largest possible value.")
    #if N > dataN.shape[0]:
    #    sys.exit("The sample number exceeds the largest possible value.")

    ### Set initial parameter point
    '''
    gamma1_ini = 0.2
    gamma2_ini = 0.2
    epsilon1_ini = 0.15
    epsilon2_ini = 0.15
    groom_ini = 0.03
    travelCost_ini = 0.2
    pushButtonCost_ini = 0.3
    '''
    parameters_iniSet = parameterMain_dict['ParaInitial']


    ### read real para from data file
    #pkl_file1 = open('para_twobox_01042018(2215).pkl', 'rb')
    #pkl_file1 = open('para_twobox_01062018(2250).pkl', 'rb')
    para_pkl = pickle.load(pkl_file1)
    pkl_file1.close()

    discount = para_pkl['discount']
    nq = para_pkl['nq']
    nr = para_pkl['nr']
    nl = para_pkl['nl']
    na = para_pkl['na']
    beta = para_pkl['foodDrop']
    gamma1 = para_pkl['appRate1']
    gamma2 = para_pkl['appRate2']
    epsilon1 = para_pkl['disappRate1']
    epsilon2 = para_pkl['disappRate1']
    rho = para_pkl['consume']
    Reward = para_pkl['reward']
    groom = para_pkl['groom']
    travelCost = para_pkl['travelCost']
    pushButtonCost = para_pkl['pushButtonCost']

    parameters = [gamma1, gamma2, epsilon1, epsilon2,
                  groom, travelCost, pushButtonCost]
    print("\nThe true parameters are", parameters)


    #### EM algorithm for parameter estimation
    print("\nEM algorithm begins ...")
    # NN denotes multiple data set, and MM denotes multiple initial points
    NN_MM_para_old_traj = []

    # NN_MM_para_new_traj = []
    NN_MM_log_likelihoods_old = []
    # NN_MM_log_likelihoods_new = []
    # NN_MM_log_likelihoods_com_old = []    # old posterior, old parameters
    # NN_MM_log_likelihoods_com_new = []    # old posterior, new parameters
    # NN_MM_latent_entropies = []

    NN_likelihoods = []

    for nn in range(NN):

        print("\nFor the", sampleIndex[nn] + 1, "-th set of data:")

        ##############################################################
        # Compute likelihood
        lat = latN[sampleIndex[nn]]
        obs = obsN[sampleIndex[nn], :, :]

        MM = len(parameters_iniSet)

        MM_para_old_traj = []
        # MM_para_new_traj = []
        MM_log_likelihoods_old = []
        # MM_log_likelihoods_new = []
        # MM_log_likelihoods_com_old = []    # old posterior, old parameters
        # MM_log_likelihoods_com_new = []    # old posterior, new parameters
        # MM_latent_entropies = []

        for mm in range(MM):
            parameters_old = np.copy(parameters_iniSet[mm])

            print("\n", mm + 1, "-th initial estimation:", parameters_old)

            itermax = E_MAX_ITER #100  # iteration number for the EM algorithm
            eps = E_EPS   # Stopping criteria for E-step in EM algorithm

            para_old_traj = []
            #para_new_traj = []

            log_likelihoods_old = []
            #log_likelihoods_new = []
            #log_likelihoods_com_old = []  # old posterior, old parameters
            #log_likelihoods_com_new = []  # old posterior, new parameters
            #latent_entropies = []

            count_E = 0
            while count_E < itermax:

                print("The", count_E + 1, "-th iteration of the EM(G) algorithm")

                if count_E == 0:
                    parameters_old = np.copy(parameters_iniSet[mm])
                else:
                    parameters_old = np.copy(parameters_new)  # update parameters

                para_old_traj.append(parameters_old)

                print(parameters_old)

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

                #log_likelihoods_com_old.append(complete_likelihood_old)
                #latent_entropies.append(latent_entropy)
                log_likelihoods_old.append(log_likelihood)

                #print(parameters_old)
                print(complete_likelihood_old)
                print(log_likelihood)

                ## Check convergence
                if len(log_likelihoods_old) >= 2 and np.abs(log_likelihood - log_likelihoods_old[-2]) < eps:
                    print("EM has converged!")
                    break

                ##########  M(G)-step ##########

                count_M = 0
                vinitial = 0
                #para_new_traj.append([])
                #log_likelihoods_com_new.append([])
                #log_likelihoods_new.append([])
                learnrate_ini = M_LR_INI
                learnrate = learnrate_ini

                # Start the gradient descent from the old parameters
                parameters_new = np.copy(parameters_old)
                complete_likelihood_new = complete_likelihood_old
                likelihood = complete_likelihood_new + latent_entropy

                #para_new_traj[count_E].append(parameters_new)
                #log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                #log_likelihoods_new[count_E].append(likelihood)

                print("    M-step")
                print("     ", parameters_new)
                print("     ", complete_likelihood_new)
                print("     ", likelihood)

                while True:

                    if parameterMain_dict['optimizer'] == 'GD':
                        ## standard gradient descent algorithm
                        #start = timeit.timeit()
                        derivative_value = twoboxGra.dQauxdpara(obs, parameters_new,
                                                                vinitial)  # vinitial is value from previous iteration, this is for computational efficiency
                        #print(timeit.timeit() - start)
                        para_temp = parameters_new + learnrate * np.array(derivative_value[:-1])
                        vinitial = derivative_value[-1]  # value iteration starts with value from previous iteration

                    elif parameterMain_dict['optimizer'] == 'PGD':
                        ## Go the potential next point with PROJECTED gradient descent
                        parameters_new_pro = - np.log(1 / parameters_new - 1)  # projected back onto the whole real axis
                        sig_deri = np.exp(- parameters_new_pro) / (np.exp(-parameters_new_pro) + 1) ** 2
                        start = timeit.timeit()
                        derivative_value = twoboxGra.dQauxdpara(obs, parameters_new, vinitial)
                        print(timeit.timeit() - start)
                        para_temp = parameters_new_pro + learnrate * np.array(derivative_value[:-1]) * sig_deri
                        para_temp = 1 / (1 + np.exp(- para_temp))  # projected onto [0,1]
                        vinitial = derivative_value[-1]

                    """
                    #change only part of the parameters
                    temp = np.copy(para_temp)
                    para_temp = np.copy(parameters_old)
                    #para_temp[0] = temp[0]
                    #para_temp[1] = temp[1]
                    #para_temp[2] = temp[2]
                    #para_temp[3] = temp[3]
                    #para_temp[4] = temp[4]
                    """

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
                    if complete_likelihood_new_temp > complete_likelihood_new + GD_THRESHOLD:
                        parameters_new = np.copy(para_temp)
                        complete_likelihood_new = complete_likelihood_new_temp
                        likelihood = complete_likelihood_new + latent_entropy

                        #para_new_traj[count_E].append(parameters_new)
                        #log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                        #log_likelihoods_new[count_E].append(likelihood)

                        print("     ", parameters_new)
                        print("     ", complete_likelihood_new)
                        print("     ", likelihood)

                        count_M += 1
                    else:
                        learnrate /= 2
                        if learnrate < learnrate_ini / (2 ** LR_DEC):
                            break

                count_E += 1

            MM_para_old_traj.append(para_old_traj)  # parameter trajectories for a particular set of data
            # MM_para_new_traj.append(para_new_traj)
            MM_log_likelihoods_old.append(log_likelihoods_old)  # likelihood trajectories for a particular set of data
            # MM_log_likelihoods_new.append(log_likelihoods_new)
            # MM_log_likelihoods_com_old.append(log_likelihoods_com_old)    # old posterior, old parameters
            # MM_log_likelihoods_com_new.append(log_likelihoods_com_new)    # old posterior, new parameters
            # MM_latent_entropies.append(latent_entropies)

        NN_MM_para_old_traj.append(MM_para_old_traj)  # parameter trajectories for all data
        # NN_MM_para_new_traj.append(MM_para_new_traj)
        NN_MM_log_likelihoods_old.append(MM_log_likelihoods_old)  # likelihood trajectories for
        # NN_MM_log_likelihoods_new.append(MM_log_likelihoods_new)
        # NN_MM_log_likelihoods_com_old.append(MM_log_likelihoods_com_old)   # old posterior, old parameters
        # NN_MM_log_likelihoods_com_new.append(MM_log_likelihoods_com_new)   # old posterior, new parameters
        # NN_MM_latent_entropies.append(MM_latent_entropies)


    #### Save result data and outputs log

    ## save the running data
    Experiment_dict = {'ParameterTrajectory': NN_MM_para_old_traj,
                       'LogLikelihood': NN_MM_log_likelihoods_old}
    output = open(datestring + dataSet + str(sampleIndex).strip('[]').replace(', ', '')  + '_ExperimentResult' + '.pkl', 'wb')
    pickle.dump(Experiment_dict, output)
    output.close()

    #np.savez('ExperimentResult_npz' + datestring, NN_MM_para_old_traj, NN_MM_log_likelihoods_old)

    ## save running parameters
    # parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
    #                       'GD_THRESHOLD': GD_THRESHOLD,
    #                       'E_EPS': E_EPS,
    #                       'M_LR_INI': M_LR_INI,
    #                       'LR_DEC': LR_DEC,
    #                       'dataSet': dataSet,
    #                       'ParaInitial': parameters_iniSet}
    output1 = open(datestring + dataSet + str(sampleIndex).strip('[]').replace(', ', '') + '_ParameterMain' + '.pkl', 'wb')
    pickle.dump(parameterMain_dict, output1)
    output1.close()

    print("finish")

if __name__ == "__main__":
    main()