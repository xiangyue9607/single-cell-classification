import numpy as np
import h5py
import random
def read_data():
    file=h5py.File('normCountsBuettnerEtAl.h5','r')
    train_set_data = file['LogNcountsMmus'][:]
    name_data=file['EnsIds'][:]
    name_data=np.ndarray.tolist(name_data)
    read_txt=open('list.txt','r')
    important_position_list=[]
    for line in read_txt.readlines():
        line=line.strip('\n')
        position=name_data.index(line)
        # print position
        important_position_list.append(position)
    print important_position_list
    tag_array=[]
    for num in range(0,182):
        if(num<59):
            tag_array.append([1])
        elif(num<117):
            tag_array.append([2])
        else:
            tag_array.append([3])

    train_set_data=np.ndarray.tolist(train_set_data)
    new_train_data=[]
    for each_data in train_set_data:
        new_data = []
        for iPosition in important_position_list:
            new_data.append(each_data[iPosition])
        new_train_data.append(new_data)

    file.close()

    label=np.array(tag_array)
    sample=np.array(new_train_data)

    return (sample,label)





