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
from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import ReadDataForRNN as rdfr
import PCA as pca


# Mutiple-Layer Perception
def MLP_k_fold_croos_validation(sample, lable):
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


def MLP_leave_one_cross_validation(sample, lable):
    length = len(sample)
    right, first, second, third = 0, 0, 0, 0
    for k in range(0, length):
        nn = Classifier(
            layers=[
                Layer("ExpLin", units=1000),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=27)
        train_sample = copy.deepcopy(sample)
        lable_sample = copy.deepcopy(lable)

        test_sample = np.array([sample[k]])
        test_lable = lable[k]
        train_sample = np.delete(train_sample, k, 0)
        lable_sample = np.delete(lable_sample, k, 0)
        # train_sample.pop(k)
        # lable_sample.pop(k)
        # print train_sample.shape
        # print lable_sample.shape
        nn.fit(train_sample, lable_sample)
        # print test_sample.shape
        test_result = nn.predict(test_sample)
        print "predict_label: ", test_result[0][0]
        print "true_label: ", test_lable[0]

        if (test_lable[0] == 1):
            if (test_result[0][0] == test_lable[0]):
                print True
                first += 1
                right += 1
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
    print "class G1:", 1.0 * first / 20
    print "class S:", 1.0 * second / 7
    print "class G2:", 1.0 * third / 8
    print "class total:", 1.0 * right / 35


def SVM_k_fold_croos_validation(sample, lable):
    X = sample
    y = lable
    kf = KFold(n_splits=10, shuffle=True)
    split_num = kf.get_n_splits(X)
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


def SVM_leave_one_cross_validation(sample, lable):
    length = len(sample)
    right, first, second, third = 0, 0, 0, 0
    for k in range(0, length):
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

        train_sample = copy.deepcopy(sample)
        lable_sample = copy.deepcopy(lable)

        test_sample = np.array([sample[k]])
        test_lable = lable[k]
        train_sample = np.delete(train_sample, k, 0)
        lable_sample = np.delete(lable_sample, k, 0)
        # train_sample.pop(k)
        # lable_sample.pop(k)
        # print train_sample.shape
        # print lable_sample.shape
        clf.fit(train_sample, lable_sample)
        # print test_sample.shape
        test_result = clf.predict(test_sample)
        print "predict_label: ", test_result[0]
        print "true_label: ", test_lable[0]

        if (test_lable[0] == 1):
            if (test_result[0] == test_lable[0]):
                print True
                first += 1
                right += 1
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
            if (test_result[0] == test_lable[0]):
                print True
                third += 1
                right += 1
            else:
                print False
        print "...................................................................................................."
        print k
        print "...................................................................................................."
    print "class G1:", 1.0 * first / 59
    print "class S:", 1.0 * second / 58
    print "class G2:", 1.0 * third / 65
    print "class total:", 1.0 * right / 182


# Decision Tree
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


# Random Forest
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


# GradientBoostingClassifier
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


def RNN_leave_one_cross_validation(seq, targets, n_updates=250, n_seq=182):
    """ Test RNN with softmax outputs. """
    length = len(seq)
    right, first, second, third = 0, 0, 0, 0
    for k in range(0, length):
        n_hidden = 10
        n_in = 40
        n_classes = 3
        n_out = n_classes  # restricted to single softmax per time step

        np.random.seed(0)

        train_sample = copy.deepcopy(seq)
        train_lable = copy.deepcopy(targets)

        test_sample = seq[k]

        train_sample = np.delete(train_sample, k, 0)
        train_lable = np.delete(train_lable, k, 0)

        train_sample = [i for i in train_sample]
        train_lable = [i for i in train_lable]

        gradient_dataset = SequenceDataset([train_sample, train_lable], batch_size=None,
                                           number_batches=500)
        cg_dataset = SequenceDataset([train_sample, train_lable], batch_size=None,
                                     number_batches=100)

        model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                        activation='tanh', output_type='softmax',
                        use_symbolic_softmax=True)

        opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                           s=model.rnn.y_pred,
                           costs=[model.rnn.loss(model.y),
                                  model.rnn.errors(model.y)], h=model.rnn.h)

        opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

        guess = model.predict_proba(test_sample)
        tmp_list = np.ndarray.tolist(guess.T)
        print tmp_list
        if (targets[k][0] == 0):
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                first += 1
                right += 1
            else:
                print k, False
        elif (targets[k][0] == 1):
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                second += 1
                right += 1
            else:
                print k, False
        else:
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                third += 1
                right += 1
            else:
                print k, False
    G1_rate = 1.0 * first / 59
    S_rate = 1.0 * second / 58
    G2_rate = 1.0 * third / 65
    total_rate = 1.0 * right / length
    tmp_str = str(n_updates) + '\n' + str(G1_rate) + '\n' + str(S_rate) + "\n" + str(G2_rate) + '\n' + str(
        total_rate) + '\n\n'
    file = open('result.txt', 'a+')
    file.write(tmp_str)
    file.close()
    # print "class G1:", 1.0 * first / 59
    # print "class S:", 1.0 * second / 58
    # print "class G2:", 1.0 * third / 65
    # print "class total:", 1.0 * right / length


def RNN_k_fold_croos_validation(sample, lable, n_updates=250):
    X = sample
    y = lable
    kf = KFold(n_splits=10, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        n_hidden = 10
        n_in = 30
        n_classes = 3
        n_out = n_classes  # restricted to single softmax per time step

        np.random.seed(0)

        train_sample = [i for i in X_train]
        train_lable = [i for i in y_train]

        gradient_dataset = SequenceDataset([train_sample, train_lable], batch_size=None,
                                           number_batches=500)
        cg_dataset = SequenceDataset([train_sample, train_lable], batch_size=None,
                                     number_batches=100)

        model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                        activation='tanh', output_type='softmax',
                        use_symbolic_softmax=True)

        opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                           s=model.rnn.y_pred,
                           costs=[model.rnn.loss(model.y),
                                  model.rnn.errors(model.y)], h=model.rnn.h)

        opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, X_test.shape[0]):
            guess = model.predict_proba(X_test[count])
            tmp_list = np.ndarray.tolist(guess.T)
            # print test_sample.shape
            if (y_test[count][0] == 0):
                first_sum += 1
                if (tmp_list.index(max(tmp_list)) == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print k, False
            elif (y_test[count][0] == 1):
                second_sum += 1
                if (tmp_list.index(max(tmp_list)) == y_test[count][0]):
                    print True
                    second += 1
                    right += 1
                else:
                    print False
            else:
                third_sum += 1
                if (tmp_list.index(max(tmp_list)) == y_test[count][0]):
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
        Total = Total + 1.0 * right / X_test.shape[0]
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


def MLP_test_real(X_train, y_train,X_test,y_test):
    nn = Classifier(
        layers=[
            Layer("Rectifier", units=1000),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=27)
    G1, G2, S, Total = 0, 0, 0, 0
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
    print "...................................................................................................."
    G1 = G1 + 1.0 * first / first_sum
    S = S + 1.0 * second / second_sum
    G2 = G2 + 1.0 * third / third_sum
    Total = Total + 1.0 * right / test_result.size
    print "class G1:", G1
    print "class S:", S
    print "class G2:", G2
    print "class total:", Total



data = pca.read_37_data_for_Sklearn()
X = data[0]
Y = data[1]

data_2=pca.read_182_data_for_Sklearn()
X_2=data_2[0]
Y_2=data_2[1]

MLP_test_real(X_2,Y_2,X,Y)
# data_for_RNN=pca.read_data()
# X_R=data_for_RNN[0]
# Y_R=data_for_RNN[1]
#
# MLP_k_fold_croos_validation(X,Y)
# MLP_leave_one_cross_validation(X, Y)
# SVM_k_fold_croos_validation(X,Y)
# SVM_leave_one_cross_validation(X,Y)
# DT_k_fold_croos_validation(X,Y)
# RF_k_fold_croos_validation(X,Y)
# GBC_k_fold_croos_validation(X, Y)
# RNN_k_fold_croos_validation(X_R,Y_R,20)
# RNN_leave_one_cross_validation(X_R,Y_R,11)
