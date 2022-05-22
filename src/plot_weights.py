
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm
# 微软雅黑,如果需要宋体,可以用simsun.ttc
myfont = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

sys.path.append('./')
# plt.rcParams['font.sans-serif'] = ['FangSong']
# plt.rcParams['axes.unicode_minus']=False

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

def make_heat_map(path, begin, line_width = 0.003, patical = False):
    
    matrix = np.load(path)
    matrix = matrix.reshape((matrix.shape[1], matrix.shape[0], matrix.shape[2]))
    print(matrix.shape)
    if patical:
        plot_len = 10
    else:
        plot_len = matrix.shape[1]
    fig, ax = plt.subplots(1, matrix.shape[0], figsize = (7 * matrix.shape[0], plot_len), sharex="col")
    
    for i in range(matrix.shape[0]):
        if patical:
            sns.heatmap(matrix[i, begin:begin+plot_len, :], linewidths = line_width, ax = ax[i], square = True, annot = False)
        else:
            sns.heatmap(matrix[i], linewidths = line_width, ax = ax[i], square = True, annot = False)
        # ax[i].set_xticklabels(['移动', '敌人', '同伴', '自己'], rotation='horizontal')
        ax[i].set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], rotation='horizontal')
        ax[i].set_yticklabels(range(begin, begin+plot_len), rotation='horizontal')
        ax[i].set_ylabel('Episode Step')
        ax[i].set_xlabel('Agent ID')
        ax[i].set_title('Agent {} Attention Map'.format(i + 1))
    if patical:
        # plt.savefig("./results/multihead/{}_{}.svg".format('MMM2_att_weights', begin))
        # plt.savefig("./results/multihead/{}_{}.jpg".format('MMM2_att_weights', begin))
        plt.savefig("./results/1206/{}_{}.svg".format('MMM2_att_weights', 'graph'))
        plt.savefig("./results/1206/{}_{}.jpg".format('MMM2_att_weights', 'graph'))

        
    else:
        plt.savefig("./results/1206/{}.jpg".format('da'))

def make_pos_fig(agent_pos_path, begin):
    plot_len = 10
    agent_pos_np = np.load(agent_pos_path)
    
    for i in range(begin, begin+plot_len):
        x = agent_pos_np[i,:,0]
        y = agent_pos_np[i,:,1]
        
        fig = plt.figure()

        plt.title('Step:{}  agent pos'.format(i))

        plt.xlabel('X')
        plt.ylabel('Y')
        # own
        plt.scatter(x[0:2],y[0:2],c = 'r',marker = 's')
        plt.scatter(x[2:9],y[2:9],c = 'r',marker = '+')
        plt.scatter(x[9],y[9],c = 'r',marker = '^')
        
        # enemy
        plt.scatter(x[13:16],y[13:16],c = 'b',marker = 's')
        plt.scatter(x[10:13],y[10:13],c = 'b',marker = '+')
        plt.scatter(x[17:22],y[17:22],c = 'b',marker = '+')
        plt.scatter(x[16],y[16],c = 'b',marker = '^')
        for id in range(len(x)):
            if id < 10:
                t_id = id
            else:
                t_id = id - 10
            plt.annotate('{}'.format(t_id + 1), xy = (x[id], y[id]))#, xytext = (x[i]+0.1, y[i]+0.1))
        plt.legend(['掠夺者', '陆战队', '医疗船'], prop=myfont)
        plt.grid()    

        # plt.savefig("./results/1206/{}_step{}.svg".format('MMM2', i))
        # plt.savefig("./results/1206/{}_step{}.jpg".format('MMM2', i))
        
        plt.savefig("./results/multihead/{}_step{}.svg".format('MMM2', i))
        plt.savefig("./results/multihead/{}_step{}.jpg".format('MMM2', i))
    
    

if __name__ == "__main__":
    
    # agent_pos_np = np.load("./MMM2_multi_head_pos.npy")
    # print(agent_pos_np.shape)
    # make_heat_map("./MMM2_multi_head_weights.npy", 50, patical=True)
    # make_pos_fig("./MMM2_multi_head_pos.npy", 50)

    make_heat_map("./MMM2_graph_weights_1206.npy", 30, patical=True)
    make_pos_fig("./MMM2_graph_pos_1206.npy", 30)