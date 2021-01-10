import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
#plt.style.use('seaborn-talk')
print(plt.style.available)
def pca_variation_plot():
        df = pd.read_csv("acc_pca_variation.csv")
        font = {'family': 'Times New Roman',
                'weight': 'medium',
                'size': 18}

        #plt.rc('font', **font)
        plt.rcParams.update({'font.size': 15, 'font.weight': 'semibold'})
        nwpu = df[df['Dataset'] == 'NWPU'].reset_index(drop=True)
        nwpu = nwpu.sort_values('PCA')
        aid = df[df['Dataset'] == 'AID'].reset_index(drop=True)
        ucm = df[df['Dataset'] == 'UCM'].reset_index(drop=True)
        #pnet = df[df['Dataset'] == 'PNet'].reset_index(drop=True)
        #print(pnet)
        #pnet = pnet[pnet['train_size']==80].reset_index(drop=True)
        print(ucm)
        fig, ax1 = plt.subplots()
        ax1.plot(ucm['PCA'], ucm['Acc_Mean'], color='red', marker='o', linestyle='dashed',
                 linewidth=2, markersize=8, label="UCM")
        ax1.plot(nwpu['PCA'], nwpu['Acc_Mean'], color='blue', marker='*', linestyle='dashed',
                 linewidth=2, markersize=8, label="NWPU")
        ax1.plot(aid['PCA'], aid['Acc_Mean'], color='green', marker='p', linestyle='dashed',
                 linewidth=2, markersize=8, label="AID")
        #ax1.plot(pnet['PCA'], pnet['Acc_Mean'], color='m', marker='P', linestyle='dashed',
        #         linewidth=2, markersize=7, label="PatternNet")
        ax1.set_ylabel('Accuracy (%)', color='black',fontsize=20)
        ax1.legend(loc = 'center right',prop={'size': 14})
        ax1.grid(linestyle = ':')
        #plt.ylim((80, 98))

        ax2 = ax1.twinx()
        ax2.plot(aid['PCA'], aid['PCA_size'], color='black', marker='X', linestyle='dashed',
                 linewidth=2, markersize=10, label="PCA Model Size")
        ax2.set_ylabel('PCA Model Size (MB)', color='black',fontsize=20)
        ax2.legend(loc = 'lower right',prop={'size': 14})
        ax1.set_xlabel("Number of PCA Components",fontsize=20)
        #ax2.legend()
        #plt.grid(ls='--')
        #plt.plot(pnet['PCA'], pnet['Acc_Mean'], color='magenda', marker='^', linestyle='dashed',
        #         linewidth=2, markersize=12, label="Base Model Size")
        #plt.savefig('accVSpcamodel.png')
        #plt.ylim((0, 8))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        plt.savefig('accVSpcamodel.eps', format='eps', bbox_inches='tight', dpi=300)
        plt.show()
def percentage_relu_plot():
    df = pd.read_csv('zero_percentage_ucm.csv')
    #font = {'family': 'Times New Roman',
    #        'weight': 'medium',
    #        'size': 18}

    #plt.rc('font', **font)
    #plt.rcParams.update({'font.size': 17})
    plt.rcParams.update({'font.size': 18, 'font.weight': 'semibold'})
    plt.tick_params(direction='out', length=6, width=2, colors='black',
                   grid_color='r', grid_alpha=0.5)

    expanddf = df[df['block_type'] == 'expand']
    print(expanddf)
    expanddf = expanddf.reset_index(drop=True)
    depthdf = df[df['block_type'] == 'depthwise']
    depthdf = depthdf.reset_index(drop=True)
    print(depthdf)
    xlabels = expanddf['block_name'].values
    labels = [label.split("k")[1] for label in xlabels]

    alpha = expanddf['zero_percentage'].values/depthdf['zero_percentage'].values
    idx = alpha.argsort()[::-1]
    #print(expanddf['zero_percentage'].values[idx])
    x_idx=idx+1
    print(idx)
    #exit()

    plt.bar(labels, expanddf['zero_percentage'].values[idx], width=1, linewidth=1.2, color='aqua',edgecolor='black')
    #plt.bar(labels, depthdf['zero_percentage'], width=1, linewidth=1.2, bottom=expanddf['zero_percentage'], color='w',
    #        hatch='///', edgecolor='r')
    plt.bar(labels, depthdf['zero_percentage'].values[idx], width=0.60, linewidth=1.2, color='w',edgecolor='r',hatch='////')

    plt.legend(['preceding ReLU', 'following ReLU'], loc=2)
    plt.ylabel(r'$zeros (\%) \rightarrow$')
    plt.xlabel(r'$layer \rightarrow$')
    plt.ylim((0, 1))
    plt.xticks(rotation=90)
    plt.xticks(ticks=range(0,len(idx)),labels=x_idx)
    plt.savefig('zero_percentage_of_ucm_1.eps', format='eps', bbox_inches='tight',dpi=300)
    plt.show()
def modelSize_Accuracy():
    '''
    ##########previous code for plot######################
    plt.rcParams.update({'font.size': 15, 'font.weight': 'semibold'})
    df = pd.read_csv('modelSizevsAccuracy.csv')
    print(df)
    fig, ax1 = plt.subplots()

    ax1.plot(df['Layer'], df['Total Size'], linestyle='--', marker='o',linewidth=2,markersize=10, label="Model Size",color='r')
    ax2=ax1.twinx()
    ax2.plot(df['Accuracy Mean'],linestyle='--', marker='X',linewidth=2,markersize=10, label="Accuracy",color='b')
    #ax1.tick_params(labelrotation=45)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    ax1.legend(loc = 'lower left',prop={'size': 14},bbox_to_anchor=(0.0, 1, 0.2, 0.5))
    ax2.legend(loc = 'lower right',prop={'size': 14},bbox_to_anchor=(0.0, 1, 1, 0.5))

    ax2.set_ylabel('Accuracy (%)', color='black',fontsize=20)
    ax1.set_ylabel('Model Size (MB)', color='black',fontsize=20)
    ax1.set_xlabel('Layer Chunks',fontsize=20)
    #ax1.legend(prop={'size': 9})
    ax1.grid(linestyle = ':')
    plt.tight_layout()
    plt.savefig('modelsize_vs_accuracy.eps', format='eps', bbox_inches='tight', dpi=300)
    plt.show()
    '''
    plt.rcParams.update({'font.size': 15, 'font.weight': 'semibold'})
    df = pd.read_csv('modelSizevsAccuracy.csv')
    df = df.sort_values(by=['Accuracy Mean'], ascending=False)
    #fig = plt.figure()  # Create matplotlib figure
    fig, ax = plt.subplots()
    #ax = fig.add_subplot(111)  # Create matplotlib axes
    ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

    width = 0.4

    df['Total Size'].plot(kind='bar', color='none', ax=ax, width=width, position=1,hatch='xxxxx',edgecolor='steelblue',lw=2,label='Model Size')
    df['Accuracy Mean'].plot(kind='bar', color='none', ax=ax2, width=width, position=0,hatch='..',edgecolor='purple',lw=1,label='Accuracy')

    ax.set_ylabel('Model Size',fontsize=16)
    ax.set_xlabel('Layer Chunks',fontsize=16)
    ax2.set_ylabel('Accuracy',fontsize=16)
    #ax2.set_ylim((75,100))
    ax.set_ylim((2, 14))
    ax2.set_ylim((78, 92))
    #ax.set_xlabel(df['Layer'])
    rects = ax2.patches
    labels = df['Layer'].values
    ax.plot([i-0.2 for i in np.arange(len(labels))],df['Total Size'],marker='o',color='r',lw=2,ls='--',markersize=10)
    ax2.plot([i+0.2 for i in np.arange(len(labels))], df['Accuracy Mean'],marker='^',color='b',lw=2,ls='--',markersize=10)
    '''
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')
    '''
    ind = np.arange(6)
    print(labels)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    #ax.tick_params(axis='y',colors='red')
    #ax2.tick_params(axis='y', colors='b')
    ax.legend(loc = 'lower left',prop={'size': 14},bbox_to_anchor=(0.0, 1, 0.2, 0.5))
    ax2.legend(loc = 'lower right',prop={'size': 14},bbox_to_anchor=(0.0, 1, 1, 0.5))


    #ax.set_xticks(tuple(labels))
    plt.tight_layout()
    plt.savefig('modelsize_vs_accuracy2.eps', format='eps', bbox_inches='tight', dpi=300)

    plt.show()

def model_size_vs_accuracy_2():
    import matplotlib.cm as cm
    plt.rcParams.update({'font.size': 15, 'font.weight': 'semibold'})
    df = pd.read_csv('modelSizevsAccuracy.csv')
    df = df.sort_values(by=['Model Size'], ascending=True)
    print(df)
    print(df['Accuracy Mean'])
    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, 6))
    print(colors)
    markers = ["P", "o",'p','d','s','v']
    ax.plot(df['Model Size'], df['Accuracy Mean'], lw=2, ls='--', color='black')
    #ax.scatter(df['Model Size'],df['Accuracy Mean'],marker='o',c=colors,s = df['Accuracy Mean'].values*4)

    labels = np.arange(0,8)#df['Model Size'].values
    ax.set_xticks(np.arange(len(labels)))
    for label,x,y,k,m in zip(df['Layer'].values,df['Model Size'].values,df['Accuracy Mean'].values,colors,markers):
        ax.scatter(x, y, marker=m, c=k, s=y * 2)
        if label == '3-6-13':
            print('Yes')
            xytext = (25, -30)
        if label=='13-16':
            xytext = (0,10)
        if label=='3-6-13-16':
            xytext = (0,-20)
        if label=='9-12':
            xytext = (-12,13)
        #else:
        #    xytext=(0,10)
        if label=='1-4':
            xytext = (30,0)
        if label=='5-8':
            xytext = (0, 15)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     #c = k,


                     xytext=xytext,  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    #ax.set_xticklabels(labels, rotation=45)

    ax.set_ylim(78,93)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.set_xlabel('Model Size (MB)', fontsize=20)
    #plt.grid('-')
    fig.tight_layout()
    plt.savefig('modelsize_vs_accuracy3.eps', format='eps', bbox_inches='tight', dpi=300)
    plt.show()

def image_resolution_change(img):
    image = cv2.imread(img)
    plt.plot(img)
# dir = 'C:/Users/aa4cy/Documents/RBFF-Images'
# filenames = [img for img in glob.glob(dir+"//*.png")]
# print(filenames)
# for file in filenames:
#     img = cv2.imread(file)
#     plt.axis('off')
#     #print(dir+'//'+(file.split('\\')[1]).split('.')[0]+'.eps')
#     #exit()
#     plt.savefig(dir+'//'+(file.split('\\')[1]).split('.')[0]+'.eps',img,format='eps',dpi=300)
#     #plt.imshow(img)
#     #plt.show()
def alpha_plot():
    df = pd.read_csv('alpha_aid.csv')
    #font = {'family': 'Times New Roman',
    #        'weight': 'medium',
    #        'size': 18}

    #plt.rc('font', **font)
    plt.rcParams.update({'font.size': 17})
    plt.tick_params(direction='out', length=6, width=2, colors='black',
                   grid_color='r', grid_alpha=0.5)


    #df = df.reset_index(drop=True)
    #df['block_no'] = df['block_no'].astype(int)
    xlabels = df['block_no'].values
    labels = [label for label in xlabels]
    plt.xticks(labels)

    plt.bar(labels, df['alpha'], width=1, linewidth=1.2, color='w',edgecolor='black',hatch='xxxxx')


    #plt.legend(['preceding_ReLU', 'forwarding_ReLU'], loc=2)
    plt.ylabel('alpha')
    plt.xlabel('block')
    plt.ylim((0, 3))
    plt.savefig('alpha_aid.eps', format='eps', bbox_inches='tight',dpi=300)
    plt.show()
def two_sided_plot():
    df1 = pd.read_csv('zero_percentage_aid.csv')
    # font = {'family': 'Times New Roman',
    #        'weight': 'medium',
    #        'size': 18}

    # plt.rc('font', **font)
    plt.rcParams.update({'font.size': 14, 'font.weight': 'semibold'})

    expanddf = df1[df1['block_type'] == 'expand']

    expanddf = expanddf.reset_index(drop=True)
    print(expanddf)
    depthdf = df1[df1['block_type'] == 'depthwise']
    depthdf = depthdf.reset_index(drop=True)
    # Data
    df = pd.read_csv('alpha_aid.csv')
    #font = {'family': 'Times New Roman',
    #        'weight': 'medium',
    #        'size': 18}

    #plt.rc('font', **font)

    states = df['block_no'].values#["AK", "TX", "CA", "MT", "NM", "AZ", "NV", "CO", "OR", "WY", "MI",
              #"MN", "UT", "ID", "KS", "NE", "SD", "WA", "ND", "OK"]
    states=[int(i) for i in states]
    staff1,staff2 = expanddf['zero_percentage'].values, depthdf['zero_percentage'].values#np.array([20, 30, 40, 10, 15, 35, 18, 25, 22, 7, 12, 22, 3, 4, 5, 8,
            #          14, 28, 24, 32])
    sales = df['alpha'].values#expanddf['zero_percentage'].values#staff * (20 + 10 * np.random.random(staff.size))

    # Sort by number of sales staff
    idx = sales.argsort()

    states, staff1,staff2, sales = [np.take(x, idx) for x in [states, staff1,staff2, sales]]
    #print(states,staff1,staff2,sales)
    print(staff1/staff2)
    print(sales)

    y = np.arange(sales.size)


    fig, axes = plt.subplots(ncols=2,sharey=True)

    #plt.tick_params(direction='out', length=6, width=2, colors='black',
    #                grid_color='r', grid_alpha=0.5)
    axes[0].barh(y, staff1, height=1,color='aqua',edgecolor='black',label='preceding ReLU')
    axes[0].barh(y, staff2,height=0.5, color='none',edgecolor='r',hatch='////',label='following ReLU')
    #axes[0].barh(y, staff2, height=0.6, color='none', edgecolor='k',zorder=1,lw=2)
    axes[0].set(title=r'$zeros (\%) \rightarrow$')

    axes[1].barh(y, sales, height=1, color='w',edgecolor='b',hatch='xxxxx',label=r'$\alpha$')
    axes[1].set(title=r'$\alpha \rightarrow$')

    axes[0].invert_xaxis()
    axes[0].set(xlim=(0, 1))
    axes[0].set(ylabel="$\\leftarrow Layer \\rightarrow$")
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, 0.78, 0.2, 0.5), ncol=2)
    #axes[1].invert_xaxis()

    axes[0].set(yticks=y, yticklabels=states)
    #axes[0].arrow()

    #axes[0].set_yticklabels(labels=states,rotation=180)
    #axes[0].yticks(rotation=90)
    #axes[1].set(yticks=y, yticklabels=states)
    axes[0].yaxis.tick_right()



    #for ax in axes.flat:
    #    ax.margins(0.03)
    #    ax.grid('--')
    fig.subplots_adjust(wspace=-.9)
    fig.tight_layout()
    #fig.subplots_adjust(wspace=-.9)
    fig.savefig('zero_alpha_aid.eps',format='eps',dpi=300)
    plt.show()
#two_sided_plot()
#pca_variation_plot()
#alpha_plot()
#percentage_relu_plot()
#modelSize_Accuracy()
model_size_vs_accuracy_2()
#exit()




