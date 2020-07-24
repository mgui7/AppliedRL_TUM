import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#     0/   1/ 2/ 3/ 4/        5/   6/          7/         8/   9
# batch/term/s1/s2/s3/end_state/done/mean_reward/sum_reward/cycle

PATHS = ['2020-07-24_01-32-28_Noorder','2020-07-24_07-14-39_sparse','2020-07-24_07-31-50_Inorder']
# path = path[2]
# path = 'misc/' + path + '.csv'
# table = pd.read_csv(path, index_col=[0])
# table = pd.read_csv(path)

sns.set(color_codes=True)
ind_to_name = {0:'No order dense reward',1:'Sparse reward',2:'Inorder dense reward'}


def show_cargoes(NUM_OF_SAMPLES = 200,intv = 10):
    repo = []
    for i in range(3):
        plt.subplot(2,2,i+1)
        path = PATHS[i]
        path = 'misc/' + path + '.csv'
        table = pd.read_csv(path)

        sum_of_cargo = np.array(table.iloc[:,[2,3,4]])
        sum_of_cargo = [3 - sum(_) for _ in sum_of_cargo]
        sum_of_cargo = sum_of_cargo[:NUM_OF_SAMPLES]
        sns.lineplot(range(1,NUM_OF_SAMPLES + 1),sum_of_cargo)

        plt.title(ind_to_name[i])
        repo.append([sum(sum_of_cargo[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])
        plt.plot(range(int(intv/2),NUM_OF_SAMPLES,intv),[sum(sum_of_cargo[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])

    plt.subplot(224)
    for i in range(3):
        sns.lineplot(range(int(intv/2),NUM_OF_SAMPLES,intv),repo[i])
    plt.legend([ind_to_name[i] for i in range(3)])
    plt.suptitle('Number of cargoes received')

    plt.show()


def show_cycle(NUM_OF_SAMPLES = 500,intv = 10):
    repo = []
    for i in range(3):
        plt.subplot(2,2,i+1)
        path = PATHS[i]
        path = 'misc/' + path + '.csv'
        table = pd.read_csv(path)

        total_cycle = np.array(table.iloc[:,[9]])
        total_cycle = [_[0] for _ in total_cycle[:NUM_OF_SAMPLES]]
        sns.lineplot(range(1,NUM_OF_SAMPLES + 1),total_cycle)

        plt.title(ind_to_name[i])
        repo.append([sum(total_cycle[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])
        plt.plot(range(int(intv/2),NUM_OF_SAMPLES,intv),[sum(total_cycle[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])

    plt.subplot(224)
    for i in range(3):
        sns.lineplot(range(int(intv/2),NUM_OF_SAMPLES,intv),repo[i])
    plt.legend([ind_to_name[i] for i in range(3)])
    plt.suptitle('Number of cycles to complete the task')
    plt.show()


def show_reward(NUM_OF_SAMPLES = 500,intv = 10):
    repo = []
    for i in range(3):
        plt.subplot(2,2,i+1)
        path = PATHS[i]
        path = 'misc/' + path + '.csv'
        table = pd.read_csv(path)

        total_cycle = np.array(table.iloc[:,[7]])
        total_cycle = [_[0] for _ in total_cycle[:NUM_OF_SAMPLES]]
        sns.lineplot(range(1,NUM_OF_SAMPLES + 1),total_cycle)

        plt.title(ind_to_name[i])
        repo.append([sum(total_cycle[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])
        plt.plot(range(int(intv/2),NUM_OF_SAMPLES,intv),[sum(total_cycle[i:i+intv]) / intv for i in range(0,NUM_OF_SAMPLES,intv)])

    plt.subplot(224)
    for i in range(3):
        sns.lineplot(range(int(intv/2),NUM_OF_SAMPLES,intv),repo[i])
    plt.legend([ind_to_name[i] for i in range(3)])
    plt.suptitle('Mean Reward')
    plt.show()


# show_cargoes(500)

show_reward()
