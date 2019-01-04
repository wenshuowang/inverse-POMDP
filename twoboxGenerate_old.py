from twobox import *
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

SAMPLE_NUMBER = 10       # 1
SAMPLE_LENGTH = 5000    # 5000


def twoboxGenerate():
    #datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')   # current time used to set file name


    print("\nSet the parameters of the model... \n")
    ### Set all related parameters
    discount = 0.99  # temporal discount , used to solve the MDP with value iteration

    ''' Parameter for the 10 set of data
    nq = 5  # number of belief states per box
    nr = 2  # number of reward states
    nl = 3  # number of location states
    na = 5

    beta = 0.001 * 0  # .001     # available food dropped back into box after button press
    gamma1 = .3  # .1 / scale    # reward becomes available in box 1
    gamma2 = .3  # .1 / scale   # reward becomes available in box 2
    delta = 0.001 * 0  # .001    # animal trips, doesn't go to target location
    direct = 0.001 * 0  # .001   # animal goes right to target, skipping location 0
    epsilon1 = .1 #.01  # available food disappears from box 1
    epsilon2 = .1 #.01  # available food disappears from box 2
    rho = 1  # .999           # .999      # food in mouth is consumed
    # eta = .0001        # random diffusion of belief

    # State rewards
    Reward = 1  # reward per time step with food in mouth
    groom = 0.05  # 0.1     # location 0 reward

    # Action costs
    travelCost = 0.2  # 0.1
    pushButtonCost = 0.3  # 0.1

    '''
    ##For Arun
    nq = 5  # number of belief states per box
    nr = 2  # number of reward states
    nl = 3  # number of location states
    na = 5

    beta = 0.001 * 0  # .001     # available food dropped back into box after button press
    gamma1 = 1/25   #.3   #.1     # reward becomes available in box 1
    gamma2 = 1/15   #.001   #.1  # .1 / scale   # reward becomes available in box 2
    delta = 0.001 * 0  # .001    # animal trips, doesn't go to target location
    direct = 0.001 * 0  # .001   # animal goes right to target, skipping location 0
    epsilon1 = 0  #.1   #.01  # available food disappears from box 1
    epsilon2 = 0  #.999   #.01  # available food disappears from box 2
    rho = 1           # .999      # food in mouth is consumed
    # eta = .0001        # random diffusion of belief

    # State rewards
    Reward = 1 #3  # reward per time step with food in mouth
    groom = 0   #0.01   #0.05  # 0.1     # location 0 reward

    # Action costs
    travelCost = 0.1  # 0.1
    pushButtonCost = .2  # 0.1
    ##

    # parameters = [beta, gamma1, gamma2, delta, direct, epsilon1, epsilon2,
    #               rho, Reward, Groom, travelCost, pushButtonCost]
    parameters = [gamma1, gamma2, epsilon1, epsilon2,
                  groom, travelCost, pushButtonCost]

    ### Solving the MDP problem with given parameters

    print("Solving the belief MDP...")
    twobox = twoboxMDP(discount, nq, nr, na, nl, parameters)
    twobox.setupMDP()
    #twobox.solveMDP_op()
    #print np.max(np.abs(Q_op - twobox.Q))
    #print np.max(np.array(policy) - twobox.policy)
    twobox.solveMDP_sfm(initial_value = 0)
    print(twobox.softpolicy)

    fig  = plt.figure(figsize = (20, 2))
    ax = fig.add_subplot(111)
    policyfig =  ax.imshow(twobox.softpolicy)
    fig.colorbar(policyfig)
    plt.show()
    #print np.max(np.abs(Q_sfm - twobox.Qsfm))
    #print np.max(softpolicy - twobox.softpolicy)


    ### Gnerate data"""
    print("Generate data based on the true model...")
    T = SAMPLE_LENGTH
    N = SAMPLE_NUMBER
    twoboxdata = twoboxMDPdata(discount, nq, nr, na, nl, parameters, T, N)
    twoboxdata.dataGenerate_sfm(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)
    #twoboxdata.dataGenerate_op(belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0)
    hybrid = twoboxdata.hybrid
    action = twoboxdata.action
    location = twoboxdata.location
    belief1 = twoboxdata.belief1
    belief2 = twoboxdata.belief2
    reward = twoboxdata.reward
    trueState1 = twoboxdata.trueState1
    trueState2 = twoboxdata.trueState2

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward, location])  # includes the action and the observable states
    latN = np.dstack([belief1, belief2])
    truthN = np.dstack([trueState1, trueState2])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(datestring + '_dataN_twobox' + '.pkl', 'wb')
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
                 'pushButtonCost': pushButtonCost
                 }
    para_output = open(datestring + '_para_twobox' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()
    print(gamma1, gamma2, epsilon1, epsilon2)

    pkl_file1 = open(datestring + '_para_twobox' + '.pkl', 'rb')
    para_pkl = pickle.load(pkl_file1)
    pkl_file1.close()
    #print(para_pkl['disappRate2'])

    print('Data stored in files' )

if __name__ == "__main__":
    twoboxGenerate()