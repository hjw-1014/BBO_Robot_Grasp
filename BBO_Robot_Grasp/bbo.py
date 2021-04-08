import numpy as np
from own_env_gym import GraspEnv
import math as m

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.policy_search import *
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.distributions import GaussianDiagonalDistribution, GaussianCholeskyDistribution
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from own_policy import Own_policy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
tqdm.monitor_interval = 0


def plot_reward(J, alg):
    length = len(J)
    x = np.arange(0, length)
    plt.plot(x, J)
    plt.title(alg)
    plt.xlabel('Epochs')
    plt.ylabel('The mean of reward for each 100 episodes')
    plt.show()

def load_and_plot(val_name, title, alg_list, n_epochs, num_samples, range_min, range_max):
    max_vals = []
    min_vals = []
    mean_vals = []
    for a in range(len(alg_list)):
        cur_max = [range_min] * n_epochs
        cur_min = [range_max] * n_epochs
        cur_mean = [0] * n_epochs

        for s in range(num_samples):
            current_vals = read_j(file_name=val_name + alg_list[a] + " sample " + str(s))
            for i in range(len(current_vals)):
                cur_max[i] = max(current_vals[i], cur_max[i])
                cur_min[i] = min(current_vals[i], cur_min[i])
                cur_mean[i] += current_vals[i] / num_samples
        max_vals.append(cur_max)
        min_vals.append(cur_min)
        mean_vals.append(cur_mean)

    plot_compare_alg_samples(j_max=max_vals, j_min=min_vals, j_mean=mean_vals, alg_list=alg_list,
                             title=title,
                             ylabel='The mean '+val_name+' for each {} epochs')

def plot_mu(mu, t):
    length = len(mu)
    x = np.arange(0, length)
    plt.plot(x, mu)
    plt.title(t)
    plt.xlabel('Epochs')
    plt.ylabel('Mean postion value for each 100 episodes')
    plt.show()

def plot_compare_alg(J_all, alg_list):
    epochs = list(np.arange(1, len(J_all[0])+1))
    for i in range(len(J_all)):
        plt.plot(epochs, J_all[i])
    plt.title('Comparison of REPS for different epsilon')
    plt.xlabel('Epochs')
    plt.ylabel('The mean value of cumulative rewards for each {} episodes'.format(len(J_all[0])))
    plt.legend(alg_list)
    plt.show()

def plot_compare_alg_samples(j_max, j_min, j_mean, alg_list, title, ylabel):
    epochs = list(np.arange(1, len(j_mean[0]) + 1))
    for i in range(len(j_mean)):
        plt.plot(epochs, j_mean[i])
        plt.fill_between(epochs, j_max[i], j_min[i], alpha=0.2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel.format(len(j_mean[0])))
    plt.legend(alg_list)
    plt.show()

def save_val(file_name, jlist):
    f = open(file_name.replace(".", "_"), "a")
    f.truncate(0)
    for i in range(len(jlist)):
        f.writelines(str(jlist[i]))
        f.write("\n")
    f.close()

def read_j(file_name):
    lines = open(file_name.replace(".", "_"), "r").readlines()
    current_j = []
    for line in lines:
        current_j.append(float(line))
    return current_j

def plot_compare_param(J_all, algorithm: str, params: list):
    epochs = list(np.arange(1, len(J_all[0])+1))
    for i in range(len(J_all)):
        plt.plot(epochs, J_all[i])
    plt.title(algorithm)
    plt.xlabel('Epochs')
    plt.ylabel('The Mean value of cumulative rewards for each {} episodes'.format(len(J_all[0])))
    plt.legend(params)
    plt.show()


def plot_compare_mu(mu_all, algorithm: str, pos: str, params: list):
    epochs = list(np.arange(1, len(J_all[0])+1))
    for i in range(len(mu_all)):
        plt.plot(epochs, mu_all[i])
    title = algorithm + pos
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean postion value for each {} episodes'.format(len(J_all[0])))
    plt.legend(params)
    plt.show()


def experiment(alg, params, n_epochs, n_episodes, n_ep_per_fit):
    np.random.seed()
    print('============ start experiment ============')
    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = GraspEnv()
    print('============ mdp ============')

    # Policy
    n_weights = 6
    mu = np.array([-0.5, 0.0, 0.91, m.pi, 0, 0])
    sigma = np.asarray([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])#np.asarray([0.15, 0.15, 0.15, 0.4, 0.4, 0.4])
    policy = Own_policy()
    dist = GaussianDiagonalDistribution(mu, sigma)  # TODO: is this distribution right? Yes.
    agent = alg(mdp.info, dist, policy, **params)

    # Train
    dataset_callback = CollectDataset()  # TODO: should we also collect the dataset? Just keep this.
    core = Core(agent, mdp, callbacks_fit=[dataset_callback])
    #core = Core(agent, mdp)

    for i in range(n_epochs):
        print('================ core learn ================')
        core.learn(n_episodes=n_episodes,
                   n_episodes_per_fit=n_ep_per_fit)

        J = compute_J(dataset_callback.get(), gamma=mdp.info.gamma)
        print('J:', J)
        print('============================')
        dataset_callback.clean()  # Todo: learning curve? Done

        p = dist.get_parameters()
        print('p:', p)
        mu_0.append(p[:n_weights][0])
        mu_1.append(p[:n_weights][1])
        mu_2.append(p[:n_weights][2])
        mu_3.append(p[:n_weights][3])
        mu_4.append(p[:n_weights][4])
        mu_5.append(p[:n_weights][5])

        current_avg_sigma=(p[n_weights:][0]+p[n_weights:][1]+p[n_weights:][2]+p[n_weights:][3]+p[n_weights:][4]+p[n_weights:][5])/6
        avg_sigma.append(current_avg_sigma)

        # record learning curve of cumulative rewards
        logger.epoch_info(i+1, J=np.mean(J), mu=p[:n_weights], sigma=p[n_weights:])
        list_J.append(np.mean(J))


if __name__ == '__main__':
    # TODO: set algotrithms and paramsters her
    algs_params = [
        (RWR, {'beta': 1.0}),
        (REPS, {'eps': 1.0})
        ]
    params_list = []  # save the parameters we use to train
    alg_list = []
    alg_list.append("RWR with beta=1.0")
    alg_list.append("REPS with eps=1.0")
    n_epochs = 60
    num_samples=10
    current_alg=0
    for alg, params in algs_params:
        for i in range(num_samples):
            list_J = []
            mu_0 = []
            mu_1 = []
            mu_2 = []
            mu_3 = []
            mu_4 = []
            mu_5 = []
            avg_sigma=[]

            experiment(alg, params, n_epochs=n_epochs, n_episodes=50, n_ep_per_fit=50)
            print('========= experiment =========')
            #save file
            save_val(file_name="J"+alg_list[current_alg]+" sample "+str(i), jlist=list_J)
            save_val(file_name="X-position"+alg_list[current_alg]+ " sample " + str(i), jlist=mu_0)
            save_val(file_name="Y-position" + alg_list[current_alg] + " sample " + str(i), jlist=mu_1)
            save_val(file_name="Z-position" + alg_list[current_alg] + " sample " + str(i), jlist=mu_2)
            save_val(file_name="X-orientation" + alg_list[current_alg] + " sample " + str(i), jlist=mu_3)
            save_val(file_name="Y-orientation" + alg_list[current_alg] + " sample " + str(i), jlist=mu_4)
            save_val(file_name="Z-orientation" + alg_list[current_alg] + " sample " + str(i), jlist=mu_5)
            save_val(file_name="Mean of Sigma" + alg_list[current_alg] + " sample " + str(i), jlist=avg_sigma)
        current_alg+=1



    load_and_plot(val_name="J", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs, num_samples=num_samples, range_max=1, range_min=0)
    load_and_plot(val_name="X-position", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs, num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="Y-position", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs, num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="Z-position", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs,num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="X-orientation", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs,num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="Y-orientation", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs,num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="Z-orientation", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs,num_samples=num_samples, range_max=100, range_min=-100)
    load_and_plot(val_name="Mean of Sigma", title="Comparison of REPS and RWR", alg_list=alg_list, n_epochs=n_epochs,num_samples=num_samples, range_max=100, range_min=-100)



