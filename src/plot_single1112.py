from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import json
import copy
from smac.env import StarCraft2Env


plot_len = 204
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
path_number_ls = [743, 744, 745, 746, 747, 748, 749, 750, 751]

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
    





# config_path_ls = [
#     # "pymarl-master/results/sacred/01/"+config_name,        
#     # "pymarl-master/results/sacred/02/"+config_name,
#     # "pymarl-master/results/sacred/03/"+config_name,
#     # "pymarl-master/results/sacred/95/"+config_name,        
#     # "pymarl-master/results/sacred/96/"+config_name,
#     # "pymarl-master/results/sacred/97/"+config_name,
#     # "pymarl-master/results/sacred/04/"+config_name,        
#     # "pymarl-master/results/sacred/05/"+config_name,
#     # "pymarl-master/results/sacred/06/"+config_name,
#     # "pymarl-master/results/sacred/98/"+config_name,        
#     # "pymarl-master/results/sacred/99/"+config_name,
#     # "pymarl-master/results/sacred/100/"+config_name, 
#     # "pymarl-master/results/sacred/7/"+config_name,        
#     # "pymarl-master/results/sacred/8/"+config_name,
#     # "pymarl-master/results/sacred/15/"+config_name,
#     # "pymarl-master/results/sacred/23/"+config_name,        
#     # "pymarl-master/results/sacred/24/"+config_name,
#     # "pymarl-master/results/sacred/25/"+config_name,
#     # "pymarl-master/results/sacred/094/"+config_name,        
#     # "pymarl-master/results/sacred/095/"+config_name,
#     # "pymarl-master/results/sacred/096/"+config_name,         
#     # "pymarl-master/results/sacred/11/"+config_name,       
#     # "pymarl-master/results/sacred/12/"+config_name,
#     # "pymarl-master/results/sacred/17/"+config_name,
#     # "pymarl-master/results/sacred/20/"+config_name,        
#     # "pymarl-master/results/sacred/21/"+config_name,
#     # "pymarl-master/results/sacred/091/"+config_name,        
#     # "pymarl-master/results/sacred/092/"+config_name,
#     # "pymarl-master/results/sacred/093/"+config_name,
#     # "pymarl-master/results/sacred/22/"+config_name,         
#     # "pymarl-master/results/sacred/13/"+config_name,         
#     # "pymarl-master/results/sacred/14/"+config_name,
#     # "pymarl-master/results/sacred/16/"+config_name,
#     # "pymarl-master/results/sacred/26/"+config_name,        
#     # "pymarl-master/results/sacred/27/"+config_name,
#     # "pymarl-master/results/sacred/28/"+config_name,

#     # # 0510 MMM2
#     # "pymarl-master/results/sacred/7/"+config_name,          # MMM2 seed 01 QMIX
#     # "pymarl-master/results/sacred/8/"+config_name,          # MMM2 seed 02 QMIX
#     # "pymarl-master/results/sacred/15/"+config_name,         # MMM2 seed 03 QMIX
#     # "pymarl-master/results/sacred/167/"+config_name,         # MMM2 seed 01 Graph_QMIX
#     # "pymarl-master/results/sacred/168/"+config_name,         # MMM2 seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/169/"+config_name,         # MMM2 seed 03 Graph_QMIX
#     # "pymarl-master/results/sacred/201/"+config_name,        
#     # "pymarl-master/results/sacred/202/"+config_name,
#     # "pymarl-master/results/sacred/203/"+config_name,
#     # "pymarl-master/results/sacred/0199/"+config_name,
#     # "pymarl-master/results/sacred/0200/"+config_name,
#     # "pymarl-master/results/sacred/0201/"+config_name,  

#     # "pymarl-master/results/sacred/222/"+config_name,         # MMM2 seed 03 Graph_QMIX
#     # "pymarl-master/results/sacred/223/"+config_name,
#     # "pymarl-master/results/sacred/224/"+config_name,
#     # "pymarl-master/results/sacred/225/"+config_name,
#     # "pymarl-master/results/sacred/228/"+config_name,
#     # "pymarl-master/results/sacred/229/"+config_name,
#     # "pymarl-master/results/sacred/230/"+config_name,
#     # "pymarl-master/results/sacred/231/"+config_name,
#     # "pymarl-master/results/sacred/232/"+config_name,
#     # "pymarl-master/results/sacred/233/"+config_name,
#     # "pymarl-master/results/sacred/234/"+config_name,
#     # "pymarl-master/results/sacred/235/"+config_name,

#     # 0621 hrl MMM2
#     # "pymarl-master/results/sacred/7/"+config_name,
#     # "pymarl-master/results/sacred/8/"+config_name,
#     # "pymarl-master/results/sacred/15/"+config_name,
#     # "pymarl-master/results/sacred/351/"+config_name,
#     # "pymarl-master/results/sacred/355/"+config_name,
#     # "pymarl-master/results/sacred/356/"+config_name,




#     # 0621 hrl 8m
#     # "pymarl-master/results/sacred/13/"+config_name,
#     # "pymarl-master/results/sacred/14/"+config_name,
#     # "pymarl-master/results/sacred/16/"+config_name,
#     # "pymarl-master/results/sacred/348/"+config_name,
#     # "pymarl-master/results/sacred/349/"+config_name,
#     # "pymarl-master/results/sacred/350/"+config_name,

#     # 0622 hrl 3s_vs4z
#     # "pymarl-master/results/sacred/204/"+config_name,
#     # "pymarl-master/results/sacred/208/"+config_name,
#     # "pymarl-master/results/sacred/212/"+config_name,
#     # "pymarl-master/results/sacred/354/"+config_name,
#     # "pymarl-master/results/sacred/358/"+config_name,
#     # "pymarl-master/results/sacred/357/"+config_name,

#     # 0623 hrl 2s3z
#     # "pymarl-master/results/sacred/11/"+config_name,
#     # "pymarl-master/results/sacred/12/"+config_name,
#     # "pymarl-master/results/sacred/17/"+config_name,
#     # "pymarl-master/results/sacred/368/"+config_name,
#     # "pymarl-master/results/sacred/370/"+config_name,
#     # "pymarl-master/results/sacred/371/"+config_name,

#     # 0706 3s_vs_4z
#     # "pymarl-master/results/sacred/204/"+config_name,
#     # "pymarl-master/results/sacred/208/"+config_name,
#     # "pymarl-master/results/sacred/212/"+config_name,
#     # "pymarl-master/results/sacred/369/"+config_name,
#     # "pymarl-master/results/sacred/372/"+config_name,
#     # "pymarl-master/results/sacred/373/"+config_name,
    
#     # 0706 MMM2
#     # "pymarl-master/results/sacred/7/"+config_name,
#     # "pymarl-master/results/sacred/8/"+config_name,
#     # "pymarl-master/results/sacred/15/"+config_name,
#     # "pymarl-master/results/sacred/374/"+config_name,
#     # "pymarl-master/results/sacred/376/"+config_name, 
#     # "pymarl-master/results/sacred/377/"+config_name,                     

#     # 0706 2c 64g
#     # "pymarl-master/results/sacred/222/"+config_name,
#     # "pymarl-master/results/sacred/223/"+config_name,
#     # "pymarl-master/results/sacred/224/"+config_name,   
#     # "pymarl-master/results/sacred/378/"+config_name,
#     # "pymarl-master/results/sacred/379/"+config_name,
#     # "pymarl-master/results/sacred/380/"+config_name,
#     # 
    
#     # # 0726 3s4z
#     # "pymarl-master/results/sacred/204/"+config_name,
#     # "pymarl-master/results/sacred/429/"+config_name,
#     # "pymarl-master/results/sacred/212/"+config_name,
#     # "pymarl-master/results/sacred/422/"+config_name,
#     # "pymarl-master/results/sacred/423/"+config_name,
#     # "pymarl-master/results/sacred/424/"+config_name, 

#      # 0729 3s4z
#     # "pymarl-master/results/sacred/204/"+config_name,
#     # "pymarl-master/results/sacred/429/"+config_name,
#     # "pymarl-master/results/sacred/212/"+config_name,
#     # "pymarl-master/results/sacred/429/"+config_name,
#     # "pymarl-master/results/sacred/432/"+config_name,
#     # "pymarl-master/results/sacred/433/"+config_name, 



#     # # 0726 2c64zg
#     "pymarl-master/results/sacred/11/"+config_name,
#     "pymarl-master/results/sacred/482/"+config_name,
#     "pymarl-master/results/sacred/484/"+config_name,                      
            
# ]




# info_path_ls = [
#     # "pymarl-master/results/sacred/01/"+info_name,        # 1c3s5z QMIX
#     # "pymarl-master/results/sacred/02/"+info_name,
#     # "pymarl-master/results/sacred/03/"+info_name,
#     # "pymarl-master/results/sacred/95/"+info_name,        #1c3s5z Graph_QMIX
#     # "pymarl-master/results/sacred/96/"+info_name,
#     # "pymarl-master/results/sacred/97/"+info_name,
#     # "pymarl-master/results/sacred/04/"+info_name,        # 3s5z_vs_3s6z QMIX
#     # "pymarl-master/results/sacred/05/"+info_name,       
#     # "pymarl-master/results/sacred/06/"+info_name,
#     # "pymarl-master/results/sacred/98/"+info_name,        # 3s5z_vs_3s6z Graph_QMIX
#     # "pymarl-master/results/sacred/99/"+info_name,
#     # "pymarl-master/results/sacred/100/"+info_name, 
#     # "pymarl-master/results/sacred/7/"+info_name,          # MMM2 seed 01 QMIX
#     # "pymarl-master/results/sacred/8/"+info_name,          # MMM2 seed 02 QMIX
#     # "pymarl-master/results/sacred/15/"+info_name,         # MMM2 seed 03 QMIX
#     # "pymarl-master/results/sacred/89/"+info_name,         # MMM2 seed 01 Graph_QMIX
#     # "pymarl-master/results/sacred/90/"+info_name,         # MMM2 seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/91/"+info_name,         # MMM2 seed 03 Graph_QMIX
#     # "pymarl-master/results/sacred/094/"+info_name,         # multihead 
#     # "pymarl-master/results/sacred/095/"+info_name,       
#     # "pymarl-master/results/sacred/096/"+info_name,
#     # "pymarl-master/results/sacred/11/"+info_name,         # 2s3z seed 01 QMIX
#     # "pymarl-master/results/sacred/12/"+info_name,         # 2s3z seed 02 QMIX
#     # "pymarl-master/results/sacred/17/"+info_name,         # 2s3z seed 02 QMIX
#     # "pymarl-master/results/sacred/86/"+info_name,         # 2s3z seed 01 Graph_QMIX
#     # "pymarl-master/results/sacred/87/"+info_name,         # 2s3z seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/88/"+info_name,         # 2s3z seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/091/"+info_name,        # multihead
#     # "pymarl-master/results/sacred/092/"+info_name,
#     # "pymarl-master/results/sacred/093/"+info_name,
#     # "pymarl-master/results/sacred/13/"+info_name,         # 8m   seed 01 QMIX
#     # "pymarl-master/results/sacred/14/"+info_name,         # 8m   seed 02 QMIX
#     # "pymarl-master/results/sacred/16/"+info_name,         # 8m   seed 03 QMIX
#     # "pymarl-master/results/sacred/92/"+info_name,         # 8m   seed 01 Graph_QMIX
#     # "pymarl-master/results/sacred/93/"+info_name,         # 8m   seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/94/"+info_name,         # 8m   seed 03 Graph_QMIX
    
#     # 0510 MMM2
#     # "pymarl-master/results/sacred/7/"+info_name,          # MMM2 seed 01 QMIX
#     # "pymarl-master/results/sacred/8/"+info_name,          # MMM2 seed 02 QMIX
#     # "pymarl-master/results/sacred/15/"+info_name,         # MMM2 seed 03 QMIX
#     # "pymarl-master/results/sacred/167/"+info_name,         # MMM2 seed 01 Graph_QMIX
#     # "pymarl-master/results/sacred/168/"+info_name,         # MMM2 seed 02 Graph_QMIX
#     # "pymarl-master/results/sacred/169/"+info_name,         # MMM2 seed 03 Graph_QMIX
#     # "pymarl-master/results/sacred/201/"+info_name,        
#     # "pymarl-master/results/sacred/202/"+info_name,
#     # "pymarl-master/results/sacred/203/"+info_name,
#     # "pymarl-master/results/sacred/0199/"+info_name,
#     # "pymarl-master/results/sacred/0200/"+info_name,
#     # "pymarl-master/results/sacred/0201/"+info_name,    


#     # 2c vs 64zg 0524
#     # "pymarl-master/results/sacred/222/"+info_name,         # MMM2 seed 03 Graph_QMIX
#     # "pymarl-master/results/sacred/223/"+info_name,
#     # "pymarl-master/results/sacred/224/"+info_name,
#     # "pymarl-master/results/sacred/225/"+info_name,
#     # "pymarl-master/results/sacred/228/"+info_name,
#     # "pymarl-master/results/sacred/229/"+info_name,
#     # "pymarl-master/results/sacred/230/"+info_name,
#     # "pymarl-master/results/sacred/231/"+info_name,
#     # "pymarl-master/results/sacred/232/"+info_name,
#     # "pymarl-master/results/sacred/233/"+info_name,
#     # "pymarl-master/results/sacred/234/"+info_name,
#     # "pymarl-master/results/sacred/235/"+info_name,

#     # # 0622 hrl mmm2
#     # "pymarl-master/results/sacred/7/"+info_name,
#     # "pymarl-master/results/sacred/8/"+info_name,
#     # "pymarl-master/results/sacred/15/"+info_name,
#     # "pymarl-master/results/sacred/351/"+info_name,
#     # "pymarl-master/results/sacred/355/"+info_name,
#     # "pymarl-master/results/sacred/356/"+info_name,



#     # 0621 hrl 8m
#     # "pymarl-master/results/sacred/13/"+info_name,
#     # "pymarl-master/results/sacred/14/"+info_name,
#     # "pymarl-master/results/sacred/16/"+info_name,
#     # "pymarl-master/results/sacred/348/"+info_name,
#     # "pymarl-master/results/sacred/349/"+info_name,
#     # "pymarl-master/results/sacred/350/"+info_name,

#     # 0622 hrl 3s_vs_4z
#     # "pymarl-master/results/sacred/204/"+info_name,
#     # "pymarl-master/results/sacred/208/"+info_name,
#     # "pymarl-master/results/sacred/212/"+info_name,
#     # "pymarl-master/results/sacred/354/"+info_name,
#     # "pymarl-master/results/sacred/358/"+info_name,
#     # "pymarl-master/results/sacred/357/"+info_name,

#     # 0623 hrl 2s3z
#     # "pymarl-master/results/sacred/11/"+info_name,
#     # "pymarl-master/results/sacred/12/"+info_name,
#     # "pymarl-master/results/sacred/17/"+info_name,
#     # "pymarl-master/results/sacred/368/"+info_name,
#     # "pymarl-master/results/sacred/370/"+info_name,
#     # "pymarl-master/results/sacred/371/"+info_name,

#     # # 0706 3s_vs_4z
#     # "pymarl-master/results/sacred/204/"+info_name,
#     # "pymarl-master/results/sacred/208/"+info_name,
#     # "pymarl-master/results/sacred/212/"+info_name,
#     # "pymarl-master/results/sacred/369/"+info_name,
#     # "pymarl-master/results/sacred/372/"+info_name,
#     # "pymarl-master/results/sacred/373/"+info_name,
    

#     # 0706 MMM2
#     # "pymarl-master/results/sacred/7/"+info_name,
#     # "pymarl-master/results/sacred/8/"+info_name,
#     # "pymarl-master/results/sacred/15/"+info_name,
#     # "pymarl-master/results/sacred/374/"+info_name,
#     # "pymarl-master/results/sacred/376/"+info_name,
#     # "pymarl-master/results/sacred/377/"+info_name,
#     # 

#     # 0706 2c 64g
#     # "pymarl-master/results/sacred/222/"+info_name,
#     # "pymarl-master/results/sacred/223/"+info_name,
#     # "pymarl-master/results/sacred/224/"+info_name,  
#     # "pymarl-master/results/sacred/378/"+info_name,
#     # "pymarl-master/results/sacred/379/"+info_name,
#     # "pymarl-master/results/sacred/380/"+info_name,


#     # 0726 3s4z hrl重新修改
#     # "pymarl-master/results/sacred/204/"+info_name,
#     # "pymarl-master/results/sacred/429/"+info_name,
#     # "pymarl-master/results/sacred/212/"+info_name,
#     # "pymarl-master/results/sacred/422/"+info_name,
#     # "pymarl-master/results/sacred/423/"+info_name,
#     # "pymarl-master/results/sacred/424/"+info_name, 


#      # 0729 3s4z
#     # "pymarl-master/results/sacred/204/"+info_name,
#     # "pymarl-master/results/sacred/429/"+info_name,
#     # "pymarl-master/results/sacred/212/"+info_name,
#     # "pymarl-master/results/sacred/429/"+info_name,
#     # "pymarl-master/results/sacred/432/"+info_name,
#     # "pymarl-master/results/sacred/433/"+info_name, 

#     # 0726 2c64zg
#     "pymarl-master/results/sacred/11/"+info_name,
#     "pymarl-master/results/sacred/482/"+info_name,
#     "pymarl-master/results/sacred/484/"+info_name,  
# ]