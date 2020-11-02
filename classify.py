from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pickle as pk
from sklearn.model_selection import RepeatedStratifiedKFold
from statistics import stdev
from statistics import mean
from sklearn.svm import SVC
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--nPCA", help="directory of folder where each image in in a subfolder of its class")
parser.add_argument("--trainSize", help="choose among 10, 20, 50, 80")
args = parser.parse_args()
if args.nPCA:
    num_PCA = int(args.nPCA)
else:
    print("please enter a number of componenets of PCA. Enter 0 if PCA needs to be excluded")
    exit(0)
if args.trainSize:
    if int(args.trainSize)==10:
        n_split = 10
    if int(args.trainSize)==20 or int(args.trainSize)==80:
        n_split = 5
    if int(args.trainSize)==50:
        n_split = 2
else:
    print("please enter the block 3_6_13 if you want to choose 3,6 and 13 block from net")
    exit(0)

print('loading data...')

X = np.load('X.npy')
y = np.load('Y.npy')
print(X.shape)
print(y.shape)

rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=2, random_state=33)  # random_state fixed to reproduce the results
scores = []

best_acc = 0.5
for train_index, test_index in rskf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    if int(args.trainSize)==80 or int(args.trainSize)==50:
        x_train, x_test = X[train_index], X[test_index]  # use it for 50% or 80% training
        y_train, y_test = y[train_index], y[test_index]  # use it for 50% or 80% training
    elif int(args.trainSize)==10 or int(args.trainSize)==20:
        x_train, x_test = X[test_index], X[train_index]  # use it for 20% training
        y_train, y_test = y[test_index], y[train_index]  # use it for 20% training
    print(x_train.shape)
    print(x_test.shape)
    #############PCA###################
    if num_PCA!=0:
        pca = PCA(n_components=num_PCA, random_state=3)  # comment this if you want to exclude PCA
        pca.fit(x_train, y_train)  # comment this if you want to exclude PCA
        x_train = pca.transform(x_train)  # comment this if you want to exclude PCA
        x_test = pca.transform(x_test)  # comment this if you want to exclude PCA

    # pk.dump(pca, open("pca_aerial_ucm.pkl", "wb"))
    ###################LDA#######################
    lda = LDA()
    lda.fit(x_train, y_train)
    # lda = pk.load(open("lda_aerial.pkl", "rb"))
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)

    # clf = pk.load(open("svm_aerial.pkl", "rb"))
    #########SVM##################
    clf = OneVsRestClassifier(SVC(random_state=333))
    clf.fit(x_train, y_train)
    s = clf.score(x_test, y_test)

    if s > best_acc:
        best_acc = s
        if num_PCA!=0:
            pk.dump(pca, open("pca_aerial_ucm.pkl", "wb"))
        pk.dump(lda, open("lda_aerial_ucm.pkl", "wb"))
        pk.dump(clf, open("svm_aerial_ucm.pkl", "wb"))

    print(s * 100)
    scores.append(s * 100)
print("Mean Accuracy and std")
print(mean(scores), stdev(scores))