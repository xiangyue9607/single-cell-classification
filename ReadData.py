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

    # for count in range(0,35):
    #     # if(count>19 and count<27):
    #     #     continue
    #     data=[]
    #     data.append(new_train_data[count])
    #     data.append([tag_array[count]])
    #     total_data.append(data)

    # test_data=[]
    # train_data=total_data
    # # test_data=random.sample(train_data,5)
    # rand_list=[]
    # rand_count=0
    #
    # while(True):
    #     temp=random.randint(0,26)
    #     rand_count+=1
    #     if temp in rand_list:
    #         rand_count-=1
    #         continue
    #     else:
    #         rand_list.append(temp)
    #         test_data.append(total_data[temp])
    #         if(rand_count==5):
    #             break
    # # print train_data.__len__()
    # # rand_list=[5,7,9,23,25]
    # for list_index in range(0,len(rand_list)):
    #     list_num=rand_list[list_index]-list_index
    #     train_data.pop(list_num)
    file.close()
    # print rand_list
    # print train_data.__len__()
    # print test_data.__len__()
    label=np.array(tag_array)
    sample=np.array(new_train_data)
    # return (train_data,test_data)
    return (sample,label)





