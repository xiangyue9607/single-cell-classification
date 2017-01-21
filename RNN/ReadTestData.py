import numpy as np
import h5py
import random
def read_data(n_seq=35):
    file=h5py.File('normCounts_mESCquartz.h5','r')

    train_set_data = file['LogNcountsQuartz'][:]
    name_data=file['gene_names_all'][:]
    name_data=np.ndarray.tolist(name_data)
    read_txt=open('list.txt','r')
    important_position_list=[]
    for line in read_txt.readlines():
        line=line.strip('\n')
        position=name_data.index(line)
        # print position
        important_position_list.append(position)

    train_set_data=np.ndarray.tolist(train_set_data)
    new_train_data=[]
    for each_data in train_set_data:
        new_data = []
        for iPosition in important_position_list:
            new_data.append(each_data[iPosition])
        new_train_data.append([new_data])

    file.close()

    sample=np.array(new_train_data)

    targets = np.zeros((n_seq,1), dtype='int32')
    lable = 0
    for row_num in range(0, targets.shape[0]):
        if (row_num < 20):
            lable = 0
        elif (row_num < 27):
            lable = 1
        else:
            lable = 2
        for column_num in range(0, targets.shape[1]):
            targets[row_num][column_num] = lable

    return (sample,targets)




