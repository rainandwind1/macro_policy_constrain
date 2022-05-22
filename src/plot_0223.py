from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 35
tick_spacing = 500*1000
info_name = "info.json"
config_name = "config.json"
seed_num = 3

# path_number_ls = [204, 208, 212, 366,367,368, 374,375,376]      # 3s 4z
# path_number_ls = [222, 223, 224, 388,387,386, 391,390,389]           # 2c 64zg
# path_number_ls = [11, 1089, 1091, 1035] # 492, 498]   # 2s3z
# path_number_ls = [7, 8, 15, 363, 364, 365, 377, 378, 379]#167, 168, 169, 290, 288, 291]              # mmm2
# path_number_ls = [999, 1104]# [114, 999, 1103, 1104]   # bane   
# path_number_ls = [1200, 1201, 1202, 1110, 1111, 1112] # 5m 6m
# path_number_ls = ['8m9m01', '8m9m02', '8m9m03', 360,361,362, 383,384,385] # 8m 9m
# path_number_ls = [395, 396]
# path_number_ls = ['01','02','03', 409,410,412,413,414,415]
# path_number_ls = [453, 454,448,  452, 451, 446, 450,449,447] # 3s5z
# path_number_ls = [471, 462, 464, 473]  # 27m30m

# path_number_ls = [204, 366, 481, 482]  # 3s4z
# 464 462
# sparse reward
# path_number_ls = [295, 296, 297]    # 2s3z
# path_number_ls = [301, 302, 303]


#  泛化性实验
# path_number_ls = [459, 460, 461, 443, 444, 445, 440, 441, 442] # 8m_terrian
# path_number_ls = [456,457, 458, 433, 434, 435, 430, 431, 432] # 1c3s5z_class


# 1213新实验
# path_number_ls = [478, 479]  # 2s3z
# path_number_ls = [366,367,368, 482, 487, 488, 481, 489, 490] #, 515, 516, 517, 521, 522, 523]  # 3s4z  204, 208, 212,
# path_number_ls = [222, 223, 224, 388,387,386, 391,390,389, 562, 563, 564, 565, 566, 567, 584, 585, 586, 587, 588, 589]# 545,544, 543, 542, 541, 540]# [388, 484, 483, 506, 512] # 2c
# path_number_ls = [453, 454,448,452, 451, 446,450,449,447, 549, 550, 551, 548, 547, 546,594,593,592,575, 579, 580] # 3s5z
# path_number_ls = [7, 363, 499, 504] # mmm2
# path_number_ls = [204, 208, 212,366,367,368, 374,375,376,539, 538, 537, 536, 535, 534, 574, 573, 572, 571, 570, 569] # 3s4z
# path_number_ls = [222, 223, 224, 388,387,386, 391,390,389, 564, 563, 562,567, 566, 565, 589, 588, 587, 586, 585, 584]# [388, 484, 483, 506, 512] # 2c
# path_number_ls = ['8m9m01', '8m9m02', '8m9m03', 360,361,362, 383,384,385, 614, 613, 612, 611, 610, 609, 626, 625,624,623,622,621]

# 1c3s5z class
# path_number_ls = [456,457, 458, 433, 434, 435, 430, 431, 432] # 1c3s5z_class
# path_number_ls_1 = ['01', '02', '03', '180', '0150', '0151', '0145', '0148', '0149']
# path_number_ls_2 = [662, 661, 660, 659, 658, 657, 656, 655, 654]

# 3s4zterrian
# path_number_ls = [702, 703, 704, 705, 706, 707, 708, 709,710]

# 8m9m formation
# path_number_ls = [714, 721, 722,715,  719, 720, 716, 717, 718]

# 2c_64zg terrian
# path_number_ls = [723,730,731, 724, 728, 729, 725, 726, 727]

# path_number_ls = [733, 734, 735, 736, 737, 738, 739, 740, 741]

# 0203 重启 2c
# path_number_ls = [222, 223, 224, 783,797,798, 802, 809, 810, 832, 833, 834, 851, 843, 844]

# 3s4z
# path_number_ls = [204, 208, 212, 784, 799, 800, 803, 813, 814, 835, 836, 837, 839, 840, 841]

# 8m9m
# path_number_ls = ['8m9m01','8m9m02','8m9m03', 792, 795, 796, 804, 807, 808, 829, 830, 831, 845, 846, 847]

# MMM2
# path_number_ls = [7,8,15, 785, 793, 794, 805, 811, 815, 826, 827, 828, 848, 849, 850]

# path_number_ls = [222, 802]
# path_number_ls = ['8m9m01', 804]
# path_number_ls = [204, 803]
# path_number_ls = [7, 805, 785]
path_number_ls = ['8m9m01','8m9m02','8m9m03', 792, 795, 796, 857, 858, 859]

config_path_ls = []
info_path_ls = []
for num in path_number_ls:
    config_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + config_name)
    info_path_ls.append("pymarl-master/results/sacred/{}/".format(num) + info_name)


def getdata():
    data = []
    alg_name = []
    scen_name = []
    for idx, config_path in enumerate(config_path_ls):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if config["env_args"]["map_name"] not in scen_name:
                scen_name.append(config["env_args"]["map_name"])
            if config["name"] not in alg_name:
                alg_name.append(config["name"]) 

    info_count = 0
    for scen_i in range(len(scen_name)):
        data.append([])
        for alg_i in range(len(alg_name)):
            data[scen_i].append([])
            for seed_i in range(seed_num):
                with open(info_path_ls[info_count], 'r') as f:
                    info = json.load(f)
                    data[scen_i][alg_i].append(info["test_battle_won_mean"][:plot_len])
                info_count += 1
    return data, alg_name, scen_name

def plot_map_info(map_name, n_agents):
    for i in range(n_agents):
        data = np.load('pymarl-master/{}'.format(map_name) + '_{}.npy'.format(i), allow_pickle=True)
        
        print(data.shape)
        fig, ax = plt.subplots()
        
        colors = ['k','y']
        label = ['move', 'attack']
        marker = ['s','*']
        for c_id, color in enumerate(colors):
            need_idx = np.where(data[:,3]==c_id)[0]
            ax.scatter(data[need_idx,0],data[need_idx,1], c=color, marker=marker[c_id], label=label[c_id], alpha = 0.6)
        
        # plt.xlim((0, 50))
        # plt.ylim((0, 50))
        plt.title("{} - agent_{}".format(map_name, i))
        legend = ax.legend()
        plt.savefig("pymarl-master/results/hrl_map_fig/{}_{}.png".format(map_name, i))
        print("save success!")


def plot_figure():
    data, alg_name, scen_name = getdata()
    xdata = [10000 * i for i in range(plot_len)]
    linestyle = ['-', '--', ':', '-.', 'dashdot', 'dashed', 'solid']
    color = ['r', 'g', 'b', 'k', 'y', 'c', 'm']
    sns.set_style(style = "whitegrid")
    for scen_i in range(len(scen_name)):
        for alg_i in range(len(alg_name)):
            for seed_i in range(len(data[scen_i][alg_i])):
                for idx in range(len(data[scen_i][alg_i][seed_i])):
                    if idx < 3:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][:idx+1])
                    else:
                        data[scen_i][alg_i][seed_i][idx] = np.mean(data[scen_i][alg_i][seed_i][idx-3:idx+1])

    for scen_i in range(len(scen_name)):
        fig = plt.figure()
        for alg_i in range(len(alg_name)):
            ax = sns.tsplot(time=xdata, data=data[scen_i][alg_i], color=color[alg_i], linestyle=linestyle[alg_i], condition=alg_name[alg_i])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.axes.xaxis.set_ticks([500*1000, 1000*1000 , 1500*1000, 2000*1000]) 
        ax.axes.set_xticklabels(['500.0k', '1.0m', '1.5m', '2.0m'])
        # ax.axes.xaxis.set_ticks([2500*1000, 3000*1000 , 3500*1000, 4000*1000]) 
        # ax.axes.set_xticklabels(['2.5m', '3.0m', '3.5m', '4.0m'])
        plt.xlabel("T", fontsize=15)
        plt.ylabel("Test Win Rate %", fontsize=15)
        plt.title('{}'.format(scen_name[scen_i]), fontsize=15)
        plt.savefig('pymarl-master/results/fig/{}.jpg'.format(scen_name[scen_i]))


if __name__ == "__main__":
    plot_figure()
    # plot_map_info("2s3z", 5)