import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def create_mat_plot(mat, axis_names, title, filename, xlab, ylab, cmap='inferno', filetype='pdf'):
    '''
    :param mat: divergence matrix
    :param axis_names: axis_names
    :param title
    :param filename: where to be saved
    :return:
    '''
    if len(axis_names)==0:
        ax = sns.heatmap(mat,annot=True, cmap=cmap,fmt="d")
    else:
        ax = sns.heatmap(mat,annot=True, yticklabels=axis_names, fmt="d", xticklabels=axis_names, cmap=cmap)
    plt.title(title)
    params = {
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
    }
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.rcParams.update(params)
    plt.tight_layout()
    plt.savefig(filename + '.'+filetype)
    plt.clf()

