from sknn.mlp import Classifier, Layer
import numpy as np
import copy
import ReadData as rd
import ReadTestData as rtd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Mutiple-Layer Perception
def MLP_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=5, shuffle=True)
    split_num=kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = Classifier(
            layers=[
                Layer("Rectifier", units=1000),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=27)
        nn.fit(X_train, y_train)
        # print test_sample.shape
        test_result = nn.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 1):
                first_sum += 1
                if (test_result[count][0] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 2):
                second_sum += 1
                if (test_result[count][0] == y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (test_result[count][0] == y_test[count][0]):
                    print True
                    third += 1
                    right += 1
                else:
                    print False

        print "...................................................................................................."
        print k, "interation"
        print "...................................................................................................."
        G1 = G1 + 1.0 * first / first_sum
        S = S + 1.0 * second / second_sum
        G2 = G2 + 1.0 * third / third_sum
        Total = Total + 1.0 * right / test_result.size
        print "class G1:", G1 / k
        print "class S:", S / k
        print "class G2:", G2 / k
        print "class total:", Total / k
        k += 1
    print "...................................................................................................."
    print "Final Result:"
    print "...................................................................................................."
    print "class G1:", G1 / split_num
    print "class S:", S / split_num
    print "class G2:", G2 / split_num
    print "class total:", Total / split_num

def MLP_leave_one_cross_validation(sample,lable):


    length=len(sample)
    right,first,second,third=0,0,0,0
    for k in range(0,length):
        nn = Classifier(
            layers=[
                Layer("ExpLin", units=1000),
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
        elif (test_lable[0] == 2):
            if (test_result[0][0] == test_lable[0]):
                print True
                second += 1
                right += 1
            else:
                print False
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
    print "class total:",1.0*right/182

def SVM_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=5, shuffle=True)
    split_num=kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
        clf.fit(X_train, y_train)
        test_result = clf.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 1):
                first_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 2):
                second_sum += 1
                if (test_result[count]== y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    third += 1
                    right += 1
                else:
                    print False

        print "...................................................................................................."
        print k, "interation"
        print "...................................................................................................."
        G1 = G1 + 1.0 * first / first_sum
        S = S + 1.0 * second / second_sum
        G2 = G2 + 1.0 * third / third_sum
        Total = Total + 1.0 * right / test_result.size
        print "class G1:", G1 / k
        print "class S:", S / k
        print "class G2:", G2 / k
        print "class total:", Total / k
        k += 1
    print "...................................................................................................."
    print "Final Result:"
    print "...................................................................................................."
    print "class G1:", G1 / split_num
    print "class S:", S / split_num
    print "class G2:", G2 / split_num
    print "class total:", Total / split_num

def SVM_leave_one_cross_validation(sample,lable):

    length=len(sample)
    right,first,second,third=0,0,0,0
    for k in range(0,length):
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

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
        clf.fit(train_sample, lable_sample)
        # print test_sample.shape
        test_result=clf.predict(test_sample)
        print "predict_label: ",test_result[0]
        print "true_label: ",test_lable[0]

        if(test_lable[0]==1):
            if(test_result[0]==test_lable[0]):
                print True
                first+=1
                right+=1
            else:
                print False
        elif (test_lable[0] == 2):
            if (test_result[0] == test_lable[0]):
                print True
                second += 1
                right += 1
            else:
                print False
        else:
            if (test_result[0]== test_lable[0]):
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
    print "class total:",1.0*right/182

#Decision Tree
def DT_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=5, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
        clf.fit(X_train, y_train)
        test_result = clf.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 1):
                first_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 2):
                second_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    third += 1
                    right += 1
                else:
                    print False

        print "...................................................................................................."
        print k, "interation"
        print "...................................................................................................."
        G1 = G1 + 1.0 * first / first_sum
        S = S + 1.0 * second / second_sum
        G2 = G2 + 1.0 * third / third_sum
        Total = Total + 1.0 * right / test_result.size
        print "class G1:", G1 / k
        print "class S:", S / k
        print "class G2:", G2 / k
        print "class total:", Total / k
        k += 1
    print "...................................................................................................."
    print "Final Result:"
    print "...................................................................................................."
    print "class G1:", G1 / split_num
    print "class S:", S / split_num
    print "class G2:", G2 / split_num
    print "class total:", Total / split_num

#Random Forest
def RF_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=5, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
        clf.fit(X_train, y_train)
        test_result = clf.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 1):
                first_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 2):
                second_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    third += 1
                    right += 1
                else:
                    print False

        print "...................................................................................................."
        print k, "interation"
        print "...................................................................................................."
        G1 = G1 + 1.0 * first / first_sum
        S = S + 1.0 * second / second_sum
        G2 = G2 + 1.0 * third / third_sum
        Total = Total + 1.0 * right / test_result.size
        print "class G1:", G1 / k
        print "class S:", S / k
        print "class G2:", G2 / k
        print "class total:", Total / k
        k += 1
    print "...................................................................................................."
    print "Final Result:"
    print "...................................................................................................."
    print "class G1:", G1 / split_num
    print "class S:", S / split_num
    print "class G2:", G2 / split_num
    print "class total:", Total / split_num

#GradientBoostingClassifier
def GBC_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=5, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.001,
     max_depth=1, random_state=0)
        clf.fit(X_train, y_train)
        test_result = clf.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 1):
                first_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 2):
                second_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (test_result[count] == y_test[count][0]):
                    print True
                    third += 1
                    right += 1
                else:
                    print False

        print "...................................................................................................."
        print k, "interation"
        print "...................................................................................................."
        G1 = G1 + 1.0 * first / first_sum
        S = S + 1.0 * second / second_sum
        G2 = G2 + 1.0 * third / third_sum
        Total = Total + 1.0 * right / test_result.size
        print "class G1:", G1 / k
        print "class S:", S / k
        print "class G2:", G2 / k
        print "class total:", Total / k
        k += 1
    print "...................................................................................................."
    print "Final Result:"
    print "...................................................................................................."
    print "class G1:", G1 / split_num
    print "class S:", S / split_num
    print "class G2:", G2 / split_num
    print "class total:", Total / split_num


data = rd.read_data()
X = data[0]
Y = data[1]


# MLP_k_fold_croos_validation(X,Y)
# MLP_leave_one_cross_validation(X,Y)
# SVM_k_fold_croos_validation(X,Y)
# SVM_leave_one_cross_validation(X,Y)
# DT_k_fold_croos_validation(X,Y)
# RF_k_fold_croos_validation(X,Y)
# GBC_k_fold_croos_validation(X, Y)
