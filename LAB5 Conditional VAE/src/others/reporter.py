import os
import matplotlib.pyplot as plt

def read_result(root_path, type):
    path = os.path.join(root_path, type, 'train_record.txt')
    epochs, epochs_valid, psnr_valid = [], [], []
    loss, mse_loss, kld_loss, tf_ratio, kl_beta = [], [], [], [], []
    with open(path, 'r') as file:
        next(file)
        for line in file:
            if 'validate psnr' not in line:
                line = line.strip('\n').split('|')
                for i in range(len(line)):
                    line[i] = line[i].split(' ')
                epoch = int(line[0][1].strip(']'))
                epochs.append(epoch)
                loss.append(float(line[0][-2]))
                mse_loss.append(float(line[1][-2]))
                kld_loss.append(float(line[2][-2]))
                tf_ratio.append(float(line[3][-2]))
                kl_beta.append(float(line[4][-1]))
            else:
                line = line.strip('\n').strip('=').split(' ')
                epochs_valid.append(epoch)
                psnr_valid.append(float(line[-2]))
    return [epochs, epochs_valid], \
        [psnr_valid, loss, mse_loss, kld_loss, tf_ratio, kl_beta]


def plot_result(type, epochs, result, save_flag=False):
    epoch, _ = epochs
    _, loss, mse_loss, kld_loss, tf_ratio, kl_beta = result

    """plot loss, tf ratio, kl beta"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("{} KL Annealing".format(type.capitalize()))
    plot1 = ax.plot(epoch, loss, color="g", label="Total Loss")
    plot2 = ax.plot(epoch, mse_loss, color="orange",label="MSE Loss")
    plot3 = ax.plot(epoch, kld_loss, color="b", label="KLD Loss")
    # different y-axis
    ax2 = ax.twinx()
    plot4 = ax2.plot(epoch, tf_ratio, color="darkviolet", linestyle = 'dashed', label="Teacher Forcing Ratio")
    plot5 = ax2.plot(epoch, kl_beta, color="hotpink", linestyle = 'dotted', label="KL Anneal Beta")

    plots = plot1 + plot2 + plot3 + plot4 + plot5
    labels = [x.get_label() for x in plots]
    ax.legend(plots, labels, loc='lower right')

    ax.grid()
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax2.set_ylabel("Teacher Forcing Ratio / KL Anneal Beta", fontsize=8)
    ax.set_ylim([0, 0.0265])
    ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02, 0.025])
    plt.show()
    if save_flag == True:
        plt.savefig("../result/{}.png".format(type))
    plt.clf()


def plot_psnr(cyclical_psnr, monotonic_psnr, save_flag=False):
    """plot psnr comparison"""
    plt.title("Learning Curves of PSNR", fontsize=10)
    plt.plot(cyclical_psnr[0], cyclical_psnr[1], color="orange", label="Cyclical")
    plt.plot(monotonic_psnr[0], monotonic_psnr[1], color="blue", label="Monotonic")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.show()
    if save_flag == True:
        plt.savefig("../result/PSNR.png")
    plt.clf()




if __name__=="__main__":
    root_path = '../logs/fp/'
    cyclical_psnr, monotonic_psnr = None, None
    # epochs=[epochs, epochs_valid], result=[psnr_valid, loss, mse_loss, kld_loss, tf_ratio, kl_beta]
    for type in ['cyclical', 'monotonic']:
        epochs, result = read_result(root_path, type)
        plot_result(type, epochs, result, save_flag=True)
        if type == 'cyclical':
            cyclical_psnr = [epochs[1], result[0]]
        elif type == 'monotonic':
            monotonic_psnr = [epochs[1], result[0]]
    plot_psnr(cyclical_psnr, monotonic_psnr, save_flag=True)
