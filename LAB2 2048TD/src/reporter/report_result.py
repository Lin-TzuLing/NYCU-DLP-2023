import os
import numpy as np
import matplotlib.pyplot as plt

def read_result(path, max_episode):
    mean_result, max_result, winRate2048_result = [], [], []
    file = open(path, "r")
    lines = file.read().split('\n')[2:-1]
    for i in range(max_episode):
        line = lines[i].split('\t')
        # only consider episode = (1,000*N)
        if int(line[0]) % 1000 == 0:
            mean_result.append(float(line[1]))
            max_result.append(float(line[2]))
            winRate2048_result.append(float(line[3]))
    return mean_result, max_result, winRate2048_result

def plot_original(type, save_root, mean_result, max_result, winRate2048_result):
    """plot single type, (mean, max and winRate2048) learning curve"""
    episode = np.arange(0, len(mean_result)*1000, step=1000)

    plt.title("Mean (N-tuple type="+type+")", fontsize=16)
    plt.xlabel("Episode")
    plt.plot(episode, mean_result, 'blue')
    plt.savefig(save_root + "{}_{}.png".format(type,'Mean'))
    plt.show()
    plt.clf()

    plt.title("Max (N-tuple type=" + type + ")", fontsize=16)
    plt.xlabel("Episode")
    plt.plot(episode, max_result, 'orange')
    plt.savefig(save_root + "{}_{}.png".format(type, 'Max'))
    plt.show()
    plt.clf()

    plt.title("Win Rate 2048 (N-tuple type=" + type + ")", fontsize=16)
    plt.xlabel("Episode")
    plt.plot(episode, winRate2048_result, 'hotpink')
    plt.yticks(np.arange(0, 110, step=10))
    plt.savefig(save_root + "{}_{}.png".format(type, 'WinRate'))
    plt.show()
    plt.clf()


def plot_train_comparison(types, save_root, mean_dict, max_dict, winRate2048_dict):
    """plot training comparison figures (mean, max, winRate2048)"""
    episode = np.arange(0, len(mean_result) * 1000, step=1000)
    colors = ['steelblue', 'darkviolet', 'gold', 'red', 'yellowgreen']

    """mean learning curve comparison"""
    plt.title('Mean comparison', fontsize=16)
    plt.xlabel("Episode")
    for i in range(len(types)):
        type = types[i]
        plt.plot(episode, mean_dict[type], color=colors[i], label=type)

    plt.legend(loc='lower right')
    plt.savefig(save_root + "learn_comparison_{}.png".format('mean'))
    plt.show()
    plt.clf()

    """max learning curve comparison"""
    plt.title('Max comparison', fontsize=16)
    plt.xlabel("Episode")
    for i in range(len(types)):
        type = types[i]
        plt.plot(episode, max_dict[type], color=colors[i], label=type)

    plt.legend(loc='lower right')
    plt.savefig(save_root + "learn_comparison_{}.png".format('max'))
    plt.show()
    plt.clf()

    """winRate2048 learning curve comparison"""
    plt.title('winRate2048 comparison', fontsize=16)
    plt.xlabel("Episode")
    for i in range(len(types)):
        type = types[i]
        plt.plot(episode, winRate2048_dict[type], color=colors[i], label=type)

    plt.legend(loc='lower right')
    plt.savefig(save_root + "learn_comparison_{}.png".format('winRate2048'))
    plt.show()
    plt.clf()


def summarize_demo():
    print()



if __name__=="__main__":
    types = ["original_0.1", "type1_0.1", "type2_0.1", "type3_0.1", "type4_0.1"]
    max_episode = 420000
    mean_dict, max_dict, winRate2048_dict = {}, {}, {}

    for type in types:
        path = os.path.join("../result/", type+".txt")
        save_root = "../reporter/"
        mean_result, max_result, winRate2048_result = read_result(path, max_episode)
        if type=="original_0.1":
            plot_original(type, save_root, mean_result, max_result, winRate2048_result)
        print("type = {}, max 2048 win rate = {}%".format(type, max(winRate2048_result)))
        mean_dict[type] = mean_result
        max_dict[type] = max_result
        winRate2048_dict[type] = winRate2048_result

    plot_train_comparison(types, save_root, mean_dict, max_dict, winRate2048_dict)
    print('plot done')