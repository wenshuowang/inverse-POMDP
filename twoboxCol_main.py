from twoboxCol import *
from HMMtwoboxCol import *
import pickle
import sys
from datetime import datetime

E_MAX_ITER = 10       # 100    # maximum number of iterations of E-step
GD_THRESHOLD = 0.1   # 0.01      # stopping criteria of M-step (gradient descent)
E_EPS = 10 ** -8                  # stopping criteria of E-step
M_LR_INI = 5 * 10 ** -8           # initial learning rate in the gradient descent step
LR_DEC =  2                       # number of times that the learning rate can be reduced


def twoboxColGenerate(parameters, sample_length, sample_number, nq, nr = 2, nl = 3, na = 5, discount = 0.99):
    # datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

    print("\nSet the parameters of the model... \n")

    ### Set all related parameters
    #     beta = parameters[0]
    #     gamma1 = parameters[1]
    #     gamma2 = parameters[2]
    #     delta =  parameters[3]
    #     direct = parameters[4]
    #     epsilon1 = parameters[5]
    #     epsilon2 = parameters[6]
    #     rho = parameters[7]
    #     # State rewards
    #     Reward = 1
    #     groom = parameters[8]
    #     # Action costs
    #     travelCost = parameters[9]
    #     pushButtonCost = parameters[10]

    beta = 0  # available food dropped back into box after button press
    gamma1 = parameters[0]  # reward becomes available in box 1
    gamma2 = parameters[1]  # reward becomes available in box 2
    delta = 0  # animal trips, doesn't go to target location
    direct = 0  # animal goes right to target, skipping location 0
    epsilon1 = parameters[2]  # available food disappears from box 1
    epsilon2 = parameters[3]  # available food disappears from box 2
    rho = 1  # food in mouth is consumed
    # State rewards
    Reward = 1  # reward per time step with food in mouth
    groom = parameters[4]  # location 0 reward
    # Action costs
    travelCost = parameters[5]
    pushButtonCost = parameters[6]

    NumCol = np.rint(parameters[7]).astype(int)  # number of colors
    Ncol = NumCol - 1  # max value of color
    qmin = parameters[8]
    qmax = parameters[9]

    # parameters = [gamma1, gamma2, epsilon1, epsilon2,
    #              groom, travelCost, pushButtonCost, NumCol, qmin, qmax]

    ### Solving the MDP problem with given parameters
    print("Solving the belief MDP...")
    twoboxCol = twoboxColMDP(discount, nq, nr, na, nl, parameters)
    twoboxCol.setupMDP()
    twoboxCol.solveMDP_sfm(initial_value=0)


    ### Gnerate data"""
    print("Generate data based on the true model...")
    T = sample_length
    N = sample_number
    twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parameters, T, N)
    twoboxColdata.dataGenerate_sfm(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)
    # twoboxdata.dataGenerate_op(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)
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

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward, location, color1, color2])  # includes the action and the observable states
    latN = np.dstack([belief1, belief2])
    truthN = np.dstack([trueState1, trueState2])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(datestring + '_dataN_twoboxCol' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'foodDrop': beta,
                 'appRate1': gamma1,
                 'appRate2': gamma2,
                 'disappRate1': epsilon1,
                 'disappRate2': epsilon2,
                 'consume': rho,
                 'reward': Reward,
                 'groom': groom,
                 'travelCost': travelCost,
                 'pushButtonCost': pushButtonCost,
                 'ColorNumber': NumCol,
                 'qmin': qmin,
                 'qmax': qmax
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(datestring + '_para_twoboxCol' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring

def main():
    # parameters = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost, NumCol, qmin, qmax]
    #parameters_gen = np.array(list(map(float, sys.argv[1].strip('[]').split(','))))
    parameters_gen = [0.1,0.1,0.01,0.01,0.05,0.2,0.3,5,0.4,0.6]

    obsN, latN, truthN, datestring = twoboxColGenerate(parameters_gen, sample_length = 5000, sample_number = 1, nq = 5)
    # sys.stdout = logger.Logger(datestring)
    # output will be both on the screen and in the log file
    # No need to manual interaction to specify parameters in the command line
    # but the log file will be written at the end of the execution.

    parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
                          'GD_THRESHOLD': GD_THRESHOLD,
                          'E_EPS': E_EPS,
                          'M_LR_INI': M_LR_INI,
                          'LR_DEC': LR_DEC,
                          # 'dataSet': sys.argv[1],
                          # 'optimizer': sys.argv[2],   # GD: standard gradient descent, PGD: projected GD
                          # 'sampleIndex': list(map(int, sys.argv[3].strip('[]').split(','))),
                          'ParaInitial': [parameters_gen]
                          # 'ParaInitial': [np.array(list(map(float, sys.argv[2].strip('[]').split(','))))]
                          # Initial parameter is a set that contains arrays of parameters, here only consider one initial point
                          }

    ### Choose which sample is used for inference
    sampleIndex = [0]
    NN = len(sampleIndex)

    ### Set initial parameter point
    parameters_iniSet = parameterMain_dict['ParaInitial']

    ### read real para from data file
    pkl_parafile = open(datestring + '_para_twoboxCol' + '.pkl', 'rb')
    para_pkl = pickle.load(pkl_parafile)
    pkl_parafile.close()

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
    NumCol = para_pkl['ColorNumber']
    qmin = para_pkl['qmin']
    qmax = para_pkl['qmax']

    parameters = [gamma1, gamma2, epsilon1, epsilon2,
                  groom, travelCost, pushButtonCost,
                  NumCol, qmin, qmax]
    print("\nThe true parameters are", parameters)

    ### EM algorithm for parameter estimation
    print("\nEM algorithm begins ...")
    # NN denotes multiple data set, and MM denotes multiple initial points
    NN_MM_para_old_traj = []
    NN_MM_para_new_traj = []
    NN_MM_log_likelihoods_old = []
    NN_MM_log_likelihoods_new = []
    NN_MM_log_likelihoods_com_old = []  # old posterior, old parameters
    NN_MM_log_likelihoods_com_new = []  # old posterior, new parameters
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
                M_thresh = GD_THRESHOLD
                count_M = 0
                #vinitial = 0
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
                    # vinitial = derivative_value[-1]  # value iteration starts with value from previous iteration

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
                    if complete_likelihood_new_temp > complete_likelihood_new + M_thresh:
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

                        # if count_M == 10 :
                        #    M_thresh = M_thresh / 4
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
    output = open(datestring + '_EM_twoboxCol' + '.pkl', 'wb')
    pickle.dump(Experiment_dict, output)
    output.close()

    ## save running parameters
    # parameterMain_dict = {'E_MAX_ITER': E_MAX_ITER,
    #                       'GD_THRESHOLD': GD_THRESHOLD,
    #                       'E_EPS': E_EPS,
    #                       'M_LR_INI': M_LR_INI,
    #                       'LR_DEC': LR_DEC,
    #                       'dataSet': dataSet,
    #                       'ParaInitial': parameters_iniSet}
    output1 = open(datestring + '_ParameterMain_twoboxCol' + '.pkl', 'wb')
    pickle.dump(parameterMain_dict, output1)
    output1.close()

    print("finish")

if __name__ == "__main__":
    main()