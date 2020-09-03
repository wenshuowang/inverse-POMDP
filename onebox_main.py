from __future__ import division
from numpy import linalg as LA
from onebox import *
import pickle
import sys
from datetime import datetime
import os


# E_MAX_ITER = 200 # 100    # maximum number of iterations of E-step
# GD_THRESHOLD = 0.01 # 0.01      # stopping criteria of M-step (gradient descent)
# E_EPS = 10 ** -6                  # stopping criteria of E-step
# M_LR_INI = 8 * 10 ** -6           # initial learning rate in the gradient descent step
# LR_DEC =  4                       # number of times that the learning rate can be reduced

def oneboxGenerate(parameters, parametersExp, sample_length, sample_number, nq, nr = 2, na = 2, discount = 0.99):
    #datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')   # current time used to set file name

    beta = parameters[0]     # available food dropped back into box after button press
    gamma = parameters[1]   # reward becomes available
    epsilon = parameters[2]   # available food disappears
    rho = parameters[3]   # .99      # food in mouth is consumed
    pushButtonCost = parameters[4]
    Reward = 1

    gamma_e = parametersExp[0]
    epsilon_e = parametersExp[1]
    #parameters = [beta, gamma, epsilon, rho, pushButtonCost]

    ### Gnerate data"""
    print("\nGenerating data...")
    T = sample_length
    N = sample_number
    oneboxdata = oneboxMDPdata(discount, nq, nr, na, parameters, parametersExp, T, N)
    oneboxdata.dataGenerate_sfm(beliefInitial = 0, rewInitial = 0)  # softmax policy

    belief = oneboxdata.belief
    action = oneboxdata.action
    reward = oneboxdata.reward
    trueState = oneboxdata.trueState

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward])  # includes the action and the observable states
    latN = np.dstack([belief])
    truthN = np.dstack([trueState])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(datestring + '_dataN_onebox' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'na': na,
                 'foodDrop': beta,
                 'appRate': gamma,
                 'disappRate': epsilon,
                 'consume': rho,
                 'reward': Reward,
                 'pushButtonCost': pushButtonCost,
                 'appRateExperiment': gamma_e,
                 'disappRateExperiment': epsilon_e
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(datestring + '_para_onebox' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring

def main():
    ##############################################
    #
    #   python -u onebox_main.py [0.2,0.3,0.1,0.9,0.6] [0.2,0.15] \([0.1,0.4,0.3,0.7,0.8]-[0.25,0.5,0.3,0.8,0.4]-[0.35,0.2,0.2,0.6,0.5]-[0.47,0.25,0.36,0.8,0.5]\) > $(date +%m%d%Y\(%H%M\))_onebox.txt &
    #
    ##############################################
    # parameters = [beta, gamma, epsilon, rho, pushButtonCost]
    #parametersAgent = np.array(list(map(float, sys.argv[1].strip('[]').split(','))))
    parametersAgent = np.array([0.01,0.4,0.2,0.99,0.6])
    #parametersExp = np.array(list(map(float, sys.argv[2].strip('[]').split(','))))
    parametersExp = np.array([0.3,0.1])

    obsN, latN, truthN, datestring = oneboxGenerate(parametersAgent, parametersExp, sample_length = 1000, sample_number = 1, nq = 5)

    # path = os.getcwd()
    # datestring = '02062019(1602)'
    # dataN_pkl_file = open(path + '/Results/' + datestring + '_dataN_onebox.pkl', 'rb')
    # dataN_pkl = pickle.load(dataN_pkl_file)
    # dataN_pkl_file.close()
    # obsN = dataN_pkl['observations']
    # latN = dataN_pkl['beliefs']
    # truthN = dataN_pkl['trueStates']


    # sys.stdout = logger.Logger(datestring)
    # output will be both on the screen and in the log file
    # No need to manual interaction to specify parameters in the command line

    parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
                          'GD_THRESHOLD': GD_THRESHOLD,
                          'E_EPS': E_EPS,
                          'M_LR_INI': M_LR_INI,
                          'LR_DEC': LR_DEC,
                          'ParaInitial': [np.array([0.01, 0.35, 0.15, 0.99, 0.4])]
                          #'ParaInitial': [np.array([0.35,0.2,0.2,0.6,0.5])]
                          #'ParaInitial': [np.array(list(map(float, i.strip('[]').split(',')))) for i in sys.argv[3].strip('()').split('-')]
                          # Initial parameter is a set that contains arrays of parameters, divided by columns(-)
                          }

    ### Choose which sample is used for inference
    sampleIndex = [0]
    NN = len(sampleIndex)

    ### Set initial parameter point
    parameters_iniSet = parameterMain_dict['ParaInitial']

    ### read real para from data file
    # pkl_parafile = open(path + '/Results/' + datestring + '_para_onebox' + '.pkl', 'rb')
    pkl_parafile = open(datestring + '_para_onebox' + '.pkl', 'rb')
    para_pkl = pickle.load(pkl_parafile)
    pkl_parafile.close()

    discount = para_pkl['discount']
    nq = para_pkl['nq']
    nr = para_pkl['nr']
    na = para_pkl['na']
    beta = para_pkl['foodDrop']
    gamma = para_pkl['appRate']
    epsilon = para_pkl['disappRate']
    rho = para_pkl['consume']
    Reward = para_pkl['reward']
    pushButtonCost = para_pkl['pushButtonCost']
    gamma_e = para_pkl['appRateExperiment']
    epsilon_e = para_pkl['disappRateExperiment']

    print("\nThe true world parameters are:", "appearing rate =",
          gamma_e, ",disappearing rate =", epsilon_e)

    parameters = [beta, gamma, epsilon, rho, pushButtonCost]
    print("\nThe internal model parameters are", parameters)
    print("beta, probability that available food dropped back into box after button press"
          "\ngamma, rate that food appears"
          "\nepsilon, rate that food disappears"
          "\nrho, food in mouth is consumed"
          "\npushButtonCost, cost of pressing the button per unit of reward")

    print("\nThe initial points for estimation are:", parameters_iniSet)
    #### EM algorithm for parameter estimation
    print("\n\nEM algorithm begins ...")
    # NN denotes multiple data set, and MM denotes multiple initial points
    NN_MM_para_old_traj = []
    NN_MM_para_new_traj = []
    NN_MM_log_likelihoods_old = []
    NN_MM_log_likelihoods_new = []
    NN_MM_log_likelihoods_com_old = []    # old posterior, old parameters
    NN_MM_log_likelihoods_com_new = []    # old posterior, new parameters
    NN_MM_latent_entropies = []


    for nn in range(NN):

        print("\nFor the", sampleIndex[nn] + 1, "-th set of data:")

        ##############################################################
        # Compute likelihood
        obs = obsN[sampleIndex[nn], :, :]

        MM = len(parameters_iniSet)

        MM_para_old_traj = []
        MM_para_new_traj = []
        MM_log_likelihoods_old = []
        MM_log_likelihoods_new = []
        MM_log_likelihoods_com_old = []    # old posterior, old parameters
        MM_log_likelihoods_com_new = []    # old posterior, new parameters
        MM_latent_entropies = []

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
                print("\nThe", count_E + 1, "-th iteration of the EM(G) algorithm")

                if count_E == 0:
                    parameters_old = np.copy(parameters_iniSet[mm])
                else:
                    parameters_old = np.copy(parameters_new)  # update parameters

                para_old_traj.append(parameters_old)

                ##########  E-step ##########

                ## Use old parameters to estimate posterior
                oneboxGra = oneboxMDPder(discount, nq, nr, na, parameters_old)
                ThA_old = oneboxGra.ThA
                softpolicy_old = oneboxGra.softpolicy
                pi = np.ones(nq) / nq
                oneHMM = HMMonebox(ThA_old, softpolicy_old, pi)

                ## Calculate likelihood of observed and complete date, and entropy of the latent sequence
                complete_likelihood_old = oneHMM.computeQaux(obs, ThA_old, softpolicy_old)
                latent_entropy = oneHMM.latent_entr(obs)
                log_likelihood = complete_likelihood_old + latent_entropy

                log_likelihoods_com_old.append(complete_likelihood_old)
                latent_entropies.append(latent_entropy)
                log_likelihoods_old.append(log_likelihood)

                print(parameters_old)
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

                learnrate_ini = M_LR_INI
                learnrate = learnrate_ini

                # Start the gradient descent from the old parameters
                parameters_new = np.copy(parameters_old)
                complete_likelihood_new = complete_likelihood_old
                likelihood = complete_likelihood_new + latent_entropy

                para_new_traj[count_E].append(parameters_new)
                log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                log_likelihoods_new[count_E].append(likelihood)

                print('\nM-step ')
                print(parameters_new)
                print(complete_likelihood_new)
                print(likelihood)

                while True:
                    #print(learnrate)
                    para_temp = parameters_new + learnrate * np.array(oneboxGra.dQauxdpara_sim(obs, parameters_new))

                    ## Check the ECDLL (old posterior, new parameters)
                    onebox_new = oneboxMDP(discount, nq, nr, na, para_temp)
                    onebox_new.setupMDP()
                    onebox_new.solveMDP_sfm()
                    ThA_new = onebox_new.ThA
                    softpolicy_new = onebox_new.softpolicy
                    complete_likelihood_new_temp = oneHMM.computeQaux(obs, ThA_new, softpolicy_new)

                    print("         ", para_temp)
                    print("         ", complete_likelihood_new_temp)

                    ## Update the parameter if the ECDLL can be improved
                    if complete_likelihood_new_temp > complete_likelihood_new  + GD_THRESHOLD:
                        parameters_new = np.copy(para_temp)
                        complete_likelihood_new = complete_likelihood_new_temp
                        likelihood = complete_likelihood_new + latent_entropy

                        para_new_traj[count_E].append(parameters_new)
                        log_likelihoods_com_new[count_E].append(complete_likelihood_new)
                        log_likelihoods_new[count_E].append(likelihood)

                        print('\n', parameters_new)
                        print(complete_likelihood_new)
                        print(likelihood)

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

    #### Save result data and outputs log

    ## save the running data
    Experiment_dict = {'ParameterTrajectory_Estep': NN_MM_para_old_traj,
                       'ParameterTrajectory_Mstep': NN_MM_para_new_traj,
                       'LogLikelihood_Estep': NN_MM_log_likelihoods_old,
                       'LogLikelihood_Mstep': NN_MM_log_likelihoods_new,
                       'Complete_LogLikelihood_Estep': NN_MM_log_likelihoods_com_old,
                       'Complete_LogLikelihood_Mstep': NN_MM_log_likelihoods_com_new,
                       'Latent_entropies': NN_MM_latent_entropies
                       }
    output = open(datestring + '_EM_onebox' + '.pkl', 'wb')
    pickle.dump(Experiment_dict, output)
    output.close()

    ## save running parameters
    # parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
    #                       'GD_THRESHOLD': GD_THRESHOLD,
    #                       'E_EPS': E_EPS,
    #                       'M_LR_INI': M_LR_INI,
    #                       'LR_DEC': LR_DEC,
    #                       'ParaInitial': parameters_iniSet}
    output1 = open(datestring + '_ParameterMain_onebox' + '.pkl', 'wb')
    pickle.dump(parameterMain_dict, output1)
    output1.close()

    print("finish")


if __name__ == "__main__":
    main()

