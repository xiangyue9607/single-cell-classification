from sknn.mlp import Classifier, Layer
import numpy as np
import copy
import ReadData as rd
import ReadTestData as rtd


def leave_one_cross_validation(sample,lable):


    length=len(sample)
    right,first,second,third=0,0,0,0
    for k in range(0,length):
        nn = Classifier(
            layers=[
                Layer("Rectifier", units=1000),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=27)
        train_sample = copy.deepcopy(sample)
        lable_sample=copy.deepcopy(lable)


        test_sample = np.array([sample[k]])
        test_lable=lable[k]
        train_sample=np.delete(train_sample,k,0)
        lable_sample=np.delete(lable_sample,k,0)
        # train_sample.pop(k)
        # lable_sample.pop(k)
        # print train_sample.shape
        # print lable_sample.shape
        nn.fit(train_sample, lable_sample)
        # print test_sample.shape
        test_result=nn.predict(test_sample)
        print "predict_label: ",test_result[0][0]
        print "true_label: ",test_lable[0]

        if(test_lable[0]==1):
            if(test_result[0][0]==test_lable[0]):
                print True
                first+=1
                right+=1
            else:
                print False
        # elif (test_lable[0] == 2):
        #     if (test_result[0][0] == test_lable[0]):
        #         print True
        #         second += 1
        #         right += 1
        #     else:
        #         print False
        else:
            if (test_result[0][0] == test_lable[0]):
                print True
                third += 1
                right += 1
            else:
                print False
        print "...................................................................................................."
        print k
        print "...................................................................................................."
    print "class G1:",1.0*first/59
    print "class S:",1.0*second/58
    print "class G2:",1.0*third/65
    print "class total:",1.0*right/124

def test_real(train_sample, train_lable,test_sample,test_lable):
    right, first, second, third = 0, 0, 0, 0
    nn = Classifier(
        layers=[
            Layer("Rectifier", units=1000),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=27)
    nn.fit(train_sample, train_lable)

    for test_num in range(0,len(test_sample)):
        test_result = nn.predict(test_sample[test_num])
        print "predict_label: ", test_result[0][0]
        print "true_label: ", test_lable[test_num][0]

        if (test_lable[test_num][0] == 1):
            if (test_result[0][0] == test_lable[test_num][0]):
                print True
                first += 1
                right += 1
            else:
                print False
        elif (test_lable[test_num][0] == 2):
            if (test_result[0][0] == test_lable[test_num][0]):
                print True
                second += 1
                right += 1
            else:
                print False
        else:
            if (test_result[0][0] == test_lable[test_num][0]):
                print True
                third += 1
                right += 1
            else:
                print False
        print "...................................................................................................."
        print test_num
        print "...................................................................................................."
    print "class G1:", 1.0 * first / 20
    print "class S:", 1.0 * second / 7
    print "class G2:", 1.0 * third / 8
    print "class total:", 1.0 * right / 35
# train_matrix = np.array([[0, 0, 0], [2, 2, 2], [2, 6, 4], [1, 3, 4]])
# lable_matrix = np.array([[1], [1], [2], [2]])


train_data=rd.read_data()
train_matrix=train_data[0]
lable_matrix=train_data[1]

test_data=rtd.read_data()
test_sample=test_data[0]
test_lable=test_data[1]



# leave_one_cross_validation(train_matrix,lable_matrix)

test_real(train_matrix,lable_matrix,test_sample,test_lable)