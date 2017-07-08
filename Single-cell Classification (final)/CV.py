from sknn.mlp import Classifier, Layer
import numpy as np
import copy
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import PCA as pca
import evaluation
import warnings


################################
#two types of cross validation for calculation the Accuracy of the classifier:
#1. K-fold cross validation (E.g: MLP_k_fold_cross_validation)
#2. leave out one cross validation (E.g: MLP_leave_one_cross_validation)
################################
# K-fold cross validation for evaluation (AUC, Precision, Recall, F-Score), E.g: RNN_Evaluation
# N types of classifier:
# 1. Mutiple-layer perception
# 2. SVM
# 3. Decision Tree
# 4. Random Forest
# 5. GradientBoostingClassifier
# 6. RNN
###############################

# Mutiple-Layer Perception
def MLP_k_fold_cross_validation(sample, lable):
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
            n_iter=40)
        nn.fit(X_train, y_train)
        # print test_sample.shape
        test_result = nn.predict(X_test)
        right, first, second, third = 0, 0, 0, 0
        first_sum, second_sum, third_sum = 0, 0, 0
        for count in range(0, test_result.size):
            if (y_test[count][0] == 0):
                first_sum += 1
                if (test_result[count][0] == y_test[count][0]):
                    print True
                    first += 1
                    right += 1
                else:
                    print False
            elif (y_test[count][0] == 1):
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
    false_list = []
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

        nn.fit(train_sample, lable_sample)

        test_result = nn.predict(test_sample)
        print "predict_label: ", test_result[0][0]
        print "true_label: ", test_lable[0]

        if (test_lable[0] == 0):
            if (test_result[0][0] == test_lable[0]):
                print True
                first += 1
                right += 1
            else:
                print False
                false_list.append(k)
        elif (test_lable[0] == 1):
            if (test_result[0][0] == test_lable[0]):
                print True
                second += 1
                right += 1
            else:
                print False
                false_list.append(k)
        else:
            if (test_result[0][0] == test_lable[0]):
                print True
                third += 1
                right += 1
            else:
                print False
                false_list.append(k)
        print "...................................................................................................."
        print k
        print "...................................................................................................."
    # G1_rate = 1.0 * first / 59
    # S_rate = 1.0 * second / 58
    # G2_rate = 1.0 * third / 65
    print "class G1:", 1.0 * first / 59
    print "class S:", 1.0 * second / 58
    print "class G2:", 1.0 * third / 65
    print "class total:", 1.0 * right / 182
    print false_list


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
                  decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
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
    false_list = []
    for k in range(0, length):
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

        train_sample = copy.deepcopy(sample)
        lable_sample = copy.deepcopy(lable)

        test_sample = np.array([sample[k]])
        test_lable = lable[k]
        train_sample = np.delete(train_sample, k, 0)
        lable_sample = np.delete(lable_sample, k, 0)
        clf.fit(train_sample, lable_sample)
        test_result = clf.predict(test_sample)
        print "predict_label: ", test_result[0]
        print "true_label: ", test_lable[0]

        if (test_lable[0] == 0):
            if (test_result[0] == test_lable[0]):
                print True
                first += 1
                right += 1
            else:
                print False
                false_list.append(k)
        elif (test_lable[0] == 1):
            if (test_result[0] == test_lable[0]):
                print True
                second += 1
                right += 1
            else:
                print False
                false_list.append(k)
        else:
            if (test_result[0] == test_lable[0]):
                print True
                third += 1
                right += 1
            else:
                print False
                false_list.append(k)
        print "...................................................................................................."
        print k
        print "...................................................................................................."
    print "class G1:", 1.0 * first / 59
    print "class S:", 1.0 * second / 58
    print "class G2:", 1.0 * third / 65
    print "class total:", 1.0 * right / 182
    print false_list

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
    false_list = []
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
        # print tmp_list
        if (targets[k][0] == 0):
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                first += 1
                right += 1
            else:
                print k, False
                false_list.append(k)
        elif (targets[k][0] == 1):
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                second += 1
                right += 1
            else:
                print k, False
                false_list.append(k)
        else:
            if (tmp_list.index(max(tmp_list)) == targets[k][0]):
                print k, True
                third += 1
                right += 1
            else:
                print k, False
                false_list.append(k)

    print "class G1:", 1.0 * first / 59
    print "class S:", 1.0 * second / 58
    print "class G2:", 1.0 * third / 65
    print "class total:", 1.0 * right / length
    print false_list

def RNN_k_fold_croos_validation(sample, lable, n_updates=20):
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
        n_hidden = 10
        n_in = 40
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

def RNN_Evaluation(sample, lable, n_hidden=10, activation_func='tanh', n_updates=20, k_fold=5):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    (AUC, p, r, f1) = (0, 0, 0, 0)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        n_in = 20
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
                        activation=activation_func, output_type='softmax',
                        use_symbolic_softmax=True)

        opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                           s=model.rnn.y_pred,
                           costs=[model.rnn.loss(model.y),
                                  model.rnn.errors(model.y)], h=model.rnn.h)

        opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(model, X_test, y_test_vector, 0.8)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k,k_fold,AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k += 1
    AUC = AUC / k_fold
    p = p / k_fold
    r = r / k_fold
    f1 = f1 / k_fold
    return AUC, p, r, f1

def MLP_Evaluation(sample, lable, n_hidden=10, activation_func='Tanh', n_updates=20, k_fold=5):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    G1, G2, S, Total = 0, 0, 0, 0
    (AUC, p, r, f1) = (0, 0, 0, 0)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = Classifier(
            layers=[
                Layer(activation_func, units=n_hidden),
                Layer(activation_func, units=n_hidden),

####################Consider mutilpe layer condition#############################

                # Layer(activation_func, units=n_hidden),
                # Layer(activation_func, units=n_hidden),
                # Layer(activation_func, units=n_hidden),
                # Layer(activation_func, units=n_hidden),
                # Layer(activation_func, units=n_hidden),
                # Layer(activation_func, units=n_hidden),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=n_updates)
        nn.fit(X_train, y_train)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(nn, X_test, y_test_vector, 0.8)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k, k_fold, AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k=k+1
    AUC = AUC / k_fold
    p = p / k_fold
    r = r / k_fold
    f1 = f1 / k_fold
    return AUC, p, r, f1

def SVM_Evaluation(sample, lable, k_fold):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    (AUC, p, r, f1) = (0, 0, 0, 0)
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
                    max_iter=-1, probability=True, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
        model.fit(X_train, y_train)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(model, X_test, y_test_vector, threshold=0.7)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k, k_fold, AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k += 1

def RF_Evaluation(sample, lable,k_fold):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    (AUC, p, r, f1) = (0, 0, 0, 0)
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = RandomForestClassifier(n_estimators=10, max_depth=None,
                                       min_samples_split=2, random_state=0)
        model.fit(X_train, y_train)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(model, X_test, y_test_vector, threshold=0.9)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k, k_fold, AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k += 1


def GBC_Evaluation(sample, lable,k_fold):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    (AUC, p, r, f1) = (0, 0, 0, 0)
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.001,
                                           max_depth=None, random_state=0, min_samples_split=2)
        model.fit(X_train, y_train)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(model, X_test, y_test_vector, threshold=0.9)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k, k_fold, AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k += 1


def DT_Evaluation(sample, lable, k_fold):
    X = sample
    y = lable
    kf = KFold(n_splits=k_fold, shuffle=True)
    split_num = kf.get_n_splits(X)
    k = 1
    (AUC, p, r, f1) = (0, 0, 0, 0)
    G1, G2, S, Total = 0, 0, 0, 0
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                       random_state=0)
        model.fit(X_train, y_train)
        y_test_vector = np.zeros((X_test.shape[0], 3), dtype='int64')
        for count in range(0, X_test.shape[0]):
            if (y_test[count][0] == 0):
                y_test_vector[count][0] = 1
            elif (y_test[count][0] == 1):
                y_test_vector[count][1] = 1
            else:
                y_test_vector[count][2] = 1

        (AUC_k, p_k, r_k, f1_k) = evaluation.evaluate(model, X_test, y_test_vector, threshold=0.9)
        print("%s / %s Iteration:AUC: %s, Prec: %s, Rec: %s, F1: %s" % (k, k_fold, AUC_k, p_k, r_k, f1_k))
        AUC = AUC + AUC_k
        p = p + p_k
        r = r + r_k
        f1 = f1 + f1_k
        print("Average: AUC: %s, Prec: %s, Rec: %s, F1: %s" % (AUC / k, p / k, r / k, f1 / k))
        k += 1




##################################
#Argument Analasis for two classifier: RNN and MLP
#################################

def argument_analasis_experiment_for_RNN():
    for i in range(1, 30):
        data_2 = pca.read_data()
        X = data_2[0]
        Y = data_2[1]
        hidden = i * 5
        print("n_hidden = %s, activation_func='tanh',n_updates=20", i * 5)
        (AUC, p, r, f1) = RNN_Evaluation(X, Y, n_hidden=hidden, activation_func='tanh', n_updates=20)
        file = open('Argument_Analasis.txt', 'a+b')
        hidden_str = str(hidden)
        resultStr = 'n_hidden = ' + hidden_str + ' activation_func=\'tanh\' AUC:' + str(AUC) + ' Presion:' + str(
            p) + ' Recall' + str(r) + ' F1:' + str(f1) + '\n'
        file.write(resultStr)
        file.close()

def argument_analasis_experiment_for_MLP():
    for i in range(1, 200):
        data_2 = pca.read_data()
        X = data_2[0]
        Y = data_2[1]
        hidden = i * 5
        print("n_hidden = %s, activation_func='Rectifier',n_updates=20", i * 5)
        (AUC, p, r, f1) = MLP_Evaluation(X, Y, n_hidden=hidden, activation_func='Rectifier', n_updates=20)
        file = open('Argument_Analasis.txt', 'a+b')
        hidden_str = str(hidden)
        resultStr = 'n_hidden = ' + hidden_str + ' activation_func=\'Rectifier\' AUC:' + str(AUC) + ' Presion:' + str(
            p) + ' Recall' + str(r) + ' F1:' + str(f1) + '\n'
        file.write(resultStr)
        file.close()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')


#Case 1: Evaluation for MLP or RNN
    # data = pca.read_data()
    # X = data[0]
    # Y = data[1]
    # # MLP_Evaluation(X,Y,n_hidden=1000)
    # RNN_Evaluation(X, Y, n_hidden=20, activation_func='tanh', n_updates=20)

#Case 2: Calculation Classifier's Accuracy for MLP or RNN
    # data = pca.read_data()
    # X = data[0]
    # Y = data[1]
    # MLP_k_fold_cross_validation(X,Y)

#Case 3: Evaluation for SVM,RF,DT,GBC
    # data = pca.read_182_data_for_Sklearn()
    # X = data[0]
    # Y = data[1]
    # SVM_Evaluation(X,Y,5)
    ## RF_Evaluation(X,Y,5)
    ## DT_Evaluation(X,Y,5)
    ## GBC_Evaluation(X,Y,5)

#Case 4: Calculation Classifier's Accuracy for SVM,RF,DT,GBC
    # data = pca.read_182_data_for_Sklearn()
    # X = data[0]
    # Y = data[1]
    # SVM_k_fold_croos_validation(X,Y)

#Case 5 : Argument Analasis:
    # argument_analasis_experiment_for_RNN()
    # argument_analasis_experiment_for_MLP()
