import numpy as np
import matplotlib.pyplot as plt
import seaborn
from mpl_toolkits.mplot3d import Axes3D


# raw_{}_agents_actions_epi_{}.npy".format("MMM2", i)
L = 20  # test num
T = 3   # horizon
A = 10 # agent num
def get_data():
    agent_data_ls = [[] for _ in range(A)]
    result = []
    for i in range(L):
        path = "3_{}_agents_actions_epi_{}.npy".format("MMM2", i)
        data_i = np.load(path)
        for a in range(A):
            for t in range(data_i.shape[1] - 4):
                # print(data_i[a,t:t+4].shape)
                agent_data_ls[a].append(data_i[a,t:t+4])
    for agent_data in agent_data_ls:
        result.append(np.stack(agent_data))
    return result
    
def plot_data(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:,0],data[:,1],data[:,2])
    plt.savefig("./test.png")
    

if __name__ == "__main__":
    data = get_data()
    plot_data(data[0])
    print(data[0].shape)