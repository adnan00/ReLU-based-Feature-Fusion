import os

model = os.stat('aerial_model.h5').st_size/(2**20)
lda = os.stat('lda_aerial_ucm.pkl').st_size/(2**20)
svm = os.stat('svm_aerial_ucm.pkl').st_size/(2**20)
pca = os.stat('pca_aerial_ucm.pkl').st_size/(2**20)
#print(model+lda+svm)
#print(model+pca+lda+svm)
print(model)
print(pca)
print(lda)
print(svm)
