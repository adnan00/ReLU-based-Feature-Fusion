import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def pca_variation_plot():
        df = pd.read_csv("acc_pca_variation.csv")
        font = {'family': 'serif',
                'weight': 'medium',
                'size': 14}

        plt.rc('font', **font)
        nwpu = df[df['Dataset'] == 'NWPU'].reset_index(drop=True)
        nwpu = nwpu.sort_values('PCA')
        aid = df[df['Dataset'] == 'AID'].reset_index(drop=True)
        ucm = df[df['Dataset'] == 'UCM'].reset_index(drop=True)
        pnet = df[df['Dataset'] == 'PNet'].reset_index(drop=True)
        #print(pnet)
        pnet = pnet[pnet['train_size']==80].reset_index(drop=True)
        print(ucm)
        fig, ax1 = plt.subplots()
        ax1.plot(ucm['PCA'], ucm['Acc_Mean'], color='red', marker='o', linestyle='dashed',
                 linewidth=2, markersize=8, label="UCM")
        ax1.plot(nwpu['PCA'], nwpu['Acc_Mean'], color='blue', marker='*', linestyle='dashed',
                 linewidth=2, markersize=8, label="NWPU")
        ax1.plot(aid['PCA'], aid['Acc_Mean'], color='green', marker='p', linestyle='dashed',
                 linewidth=2, markersize=8, label="AID")
        ax1.plot(pnet['PCA'], pnet['Acc_Mean'], color='m', marker='P', linestyle='dashed',
                 linewidth=2, markersize=7, label="PatternNet")
        ax1.set_ylabel('Accuracy (%)', color='black')
        ax1.legend(prop={'size': 9})
        ax1.grid(linestyle = ':')

        ax2 = ax1.twinx()
        ax2.plot(aid['PCA'], aid['PCA_size'], color='black', marker='X', linestyle='dashed',
                 linewidth=2, markersize=10, label="PCA Model Size")
        ax2.set_ylabel('PCA Model Size (MB)', color='black')
        ax2.legend(loc = 'lower center',prop={'size': 9})
        ax1.set_xlabel("Number of PCA Components")
        #ax2.legend()
        #plt.grid(ls='--')
        #plt.plot(pnet['PCA'], pnet['Acc_Mean'], color='magenda', marker='^', linestyle='dashed',
        #         linewidth=2, markersize=12, label="Base Model Size")
        plt.savefig('accVSpcamodel.png')
        plt.show()
def percentage_relu_plot():
    df = pd.read_csv('zero_percentage_nwpu.csv')
    font = {'family': 'serif',
            'weight': 'medium',
            'size': 14}

    plt.rc('font', **font)

    expanddf = df[df['block_type'] == 'expand']
    print(expanddf)
    expanddf = expanddf.reset_index(drop=True)
    depthdf = df[df['block_type'] == 'depthwise']
    depthdf = depthdf.reset_index(drop=True)
    print(depthdf)
    xlabels = expanddf['block_name'].values
    labels = [label.split("k")[1] for label in xlabels]
    plt.bar(labels, expanddf['zero_percentage'], width=1, linewidth=1.2, color='w', hatch='xxx', edgecolor='b')
    plt.bar(labels, depthdf['zero_percentage'], width=1, linewidth=1.2, bottom=expanddf['zero_percentage'], color='w',
            hatch='///', edgecolor='r')

    plt.legend(['preceding_ReLU', 'forwarding_ReLU'], loc=2)
    plt.ylabel('percentage of zeros')
    plt.xlabel('block')
    plt.ylim((0, 2))
    plt.savefig('nwpu_percentage_of_zeros_1.png', format='png', bbox_inches='tight')
    plt.show()

pca_variation_plot()
percentage_relu_plot()
#exit()




