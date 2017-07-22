import numpy as np
from sklearn.decomposition import PCA
import ReadData as rd
import h5py



def read_data():
    file=h5py.File('normCountsBuettnerEtAl.h5','r')
    train_set_data = file['LogNcountsMmus'][:]
    train_set_data=np.ndarray.tolist(train_set_data)
    file.close()

    sample=np.array(train_set_data)
    pca = PCA(copy=True, iterated_power='auto', n_components=30, random_state=None,
              svd_solver='full', tol=0.0, whiten=False)
    sample=pca.fit_transform(sample)
    new_train_data=[]
    sample=np.ndarray.tolist(sample)
    for each_data in sample:
        new_train_data.append([each_data])
    sample=np.array(new_train_data)

    targets = np.zeros((182, 1), dtype='int32')
    lable = 0
    for row_num in range(0, targets.shape[0]):
        if (row_num < 59):
            lable = 0
        elif (row_num < 117):
            lable = 1
        else:
            lable = 2
        for column_num in range(0, targets.shape[1]):
            targets[row_num][column_num] = lable
    return (sample,targets)




def read_182_data_for_Sklearn():
    file = h5py.File('normCountsBuettnerEtAl.h5', 'r')
    train_set_data = file['LogNcountsMmus'][:]
    train_set_data = np.ndarray.tolist(train_set_data)
    file.close()

    sample = np.array(train_set_data)
    pca = PCA(copy=True, iterated_power='auto', n_components=20, random_state=None,
              svd_solver='full', tol=0.0, whiten=False)
    sample = pca.fit_transform(sample)
    new_train_data = []
    sample = np.ndarray.tolist(sample)
    for each_data in sample:
        new_train_data.append(each_data)
    sample = np.array(new_train_data)

    tag_array=[]
    for num in range(0,182):
        if(num<59):
            tag_array.append([1])
        elif(num<117):
            tag_array.append([2])
        else:
            tag_array.append([3])

    label=np.array(tag_array)

    return (sample,label)


def read_37_data_for_Sklearn():
    file = h5py.File('normCounts_mESCquartz.h5', 'r')
    train_set_data = file['LogNcountsQuartz'][:]
    train_set_data = np.ndarray.tolist(train_set_data)
    file.close()

    sample = np.array(train_set_data)
    pca = PCA(copy=True, iterated_power='auto', n_components=20, random_state=None,
              svd_solver='full', tol=0.0, whiten=False)
    sample = pca.fit_transform(sample)
    new_train_data = []
    sample = np.ndarray.tolist(sample)
    for each_data in sample:
        new_train_data.append(each_data)
    sample = np.array(new_train_data)

    tag_array=[]

    for num in range(0,35):
        if(num<20):
            tag_array.append([1])
        elif(num<27):
            tag_array.append([2])
        else:
            tag_array.append([3])

    label=np.array(tag_array)

    return (sample,label)
