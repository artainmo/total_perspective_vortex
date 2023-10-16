import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from processing_EEGs_lib.preprocessing import preprocessing_transformation, get_data
from processing_EEGs_lib.dimensionality_reduction_algorithm import CSPTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from random import randint
import sys

def print_shape(x, y):
    rand1 = randint(0, x.shape[0]-1)
    rand2 = randint(0, x.shape[1]-1)
    print("Datas:")
    print(" x:", x.shape, "| random value: ", x[rand1][rand2])
    print(" y:", y.shape, "| random value: ", y[rand1])

def print_pipe(pipe):
    print("Pipe steps: ")
    for step in pipe.get_params()['steps']:
        print(" -", step[0], ":", step[1])

def print_score(pipe, x, y, message=":"):
    print("Score" + message)
    score = pipe.score(x,y)
    print("", score)
    return score

def _continue():
    if g_skip and input("\nYou want to continue? (y/n) : ") == "n":
        exit()
    print("")

def preprocessed_data():
    print("\033[92mPREPROCESSING\033[0m")
    x = np.array([])
    y = np.array([])
    for set_nb in [3, 4, 7, 8, 11, 12]: # By using all the datasets we get 90 examples instead of 15
        raw_data, annotations = get_data(set_nb)
        if x.size == 0:
            x, y = preprocessing_transformation(raw_data, annotations)
        else:
            _x, _y = preprocessing_transformation(raw_data, annotations)
            x = np.concatenate((x, _x), axis=0)
            y = np.concatenate((y, _y), axis=0)
        print_shape(x, y)
    return x, y

def split_data(x, y):
    # Split in 0.6 (training), 0.2 (test), 0.2 (validation)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)
    # Validation set doesn't need to be cut because sklearn classifiers make the cut themselves
    # or in the case of DecisionTrees validation sets are not even used.
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, 
                                                                    #shuffle=True)
    return x_train, x_test, y_train, y_test

def processing(x, y):
    print("\033[92mPROCESSING\033[0m")
    steps = [
                #The recommended approach is to normalize the data after splitting it into training and testing sets. 
                #The rationale behind this recommendation is to prevent any information leakage from the testing set 
                #into the training set.
                #For this dataset minmax-normalization leads to excessive overfitting as shown in examples below.
                #Normalization may fail on this dataset because all features are on the same scale either way 
                #and minmax eliminates outliers that in certain cases can be useful signals.
                # ("normalization", MinMaxScaler()), 
                #StandardScalar uses z-score normalization which normalizes without elimination of outliers.
                #It slightly improves scores compared to no normalization while minmax completely lowers scores.
                #Thus the secret to find generalizable trends in this dataset lies in the outliers.
                ("normalization", StandardScaler()), 
                #nb_components in CSP does not seem to have much impact, 4 seems best.
                #CSP biases towards overfitting which can be good or bad as shown in examples below.
                #CSP slows down the algorithm when called before each test but not when called once before all tests.
                ("dimensionality-reduction-algorithm", CSPTransformer(nb_components=4)),
            ]
    process_x_pipe = Pipeline(steps)
    print_pipe(process_x_pipe)
    x = process_x_pipe.fit_transform(x, y)
    print_shape(x, y)
    return x

def train_test(x_train, x_test, y_train, y_test, algo):
    print("\033[92mTRAIN\033[0m")
    #For value k (n_neighbors) in practice values between 1-21 are used.
    #Higher values bias towards underfitting and lower values to overfitting.
    #KNN when optimized for test set with K=21 only bring 1.7% benefit all the while scoring 0.78 on training set.
    #When not using CSP KNN completely underfits. Thus here CSP's overfitting bias is beneficial.
    #When using CSP without minmax-normalization, score improves sufficiently (train: 0.73, test: 0.7) especially when
    #also lowering K from 21 to 2 (train: 1.0, test: 0.9844)
    #Interestingly when using StandardScaler, thus z-score normalization, thus normalization that maintains outliers
    #score improves further (train: 1.0, test: 0.997)
    if algo == "KNN":
        pipe = Pipeline([("classifier", KNeighborsClassifier(n_neighbors=2))])
    #GradientBoosting leads to overfitting, score 0.99 on training set and only 1% benefit on test set.
    #When not using CSP training score becomes 0.87 and test set 0.548 which means CSP contributes 
    #negatively to overfitting here.
    #When using CSP and normalizing a second time afterwards, scoring worsens.
    #When not minmax-normalizing before CSP, scoring stays same, while when normalizing with StandardScalar the scoring
    #completely improves (train: 0.987, test: 0.963).
    #When using n_estimators=100 scoring improves further (train: 1.0, test: 0.9857) however speed slows down a lot.
    elif algo == "GradientBoosting":
        pipe = Pipeline([("classifier", GradientBoostingClassifier(n_estimators=100, validation_fraction=0.9))])
    #DesicionTree with max_depth=3 is able to reduce overfitting (train: 0.8, 0.54 over 1000 tests)
    #however test set is still only 3.2-4.6% better than complete randomness.
    #When not using CSP overfitting improves even more (train: 0.69, test: 0.58).
    #When using CSP but not minmax-normalization score becomes good enough (train: 0.83, test: 0.78).
    #This means that cutting out outliers with minmax-normalization was pathological in this dataset 
    #even if in most datasets this is beneficial.
    #StandardScalar does not impact scoring compared to no normalization. Indicating DesicionTrees are not sensitive to
    #normalization. When changing max_depth from 1 to 3, score improves (train: 1.0, test: 0.9842).
    #DecisionTrees are clearly the fastest.
    elif algo == "DecisionTree":
        pipe = Pipeline([("classifier", DecisionTreeClassifier(max_depth=3))])
    else:
        print("train.py: Error: Classification algo", algo, "not found.")
    print_pipe(pipe)
    pipe.fit(x_train, y_train)
    score_train = print_score(pipe, x_train, y_train, " training set:")
    print("\033[92mTEST\033[0m")
    score_test = print_score(pipe, x_test, y_test, " test set:")
    return score_train, score_test, pipe.named_steps['classifier']

def main(algo):
    x, y = preprocessed_data()
    x = processing(x, y)
    _continue()
    train_scores = []
    test_scores = []
    nb_tests = 6 if g_skip else 1000
    for _ in range(nb_tests):
        x_train, x_test, y_train, y_test = split_data(x, y)
        score_train, score_test, classifier = train_test(x_train, x_test, y_train, y_test, algo)
        _continue()
        train_scores.append(score_train)
        test_scores.append(score_test)
    print("\033[92mCONCLUSIONS\033[0m")
    print("average score over", nb_tests, "experiments of training set:", sum(train_scores)/len(train_scores))
    print("average score over", nb_tests, "experiments of test set:    ", sum(test_scores)/len(test_scores))
    _continue()
    print("\033[92mCROSS VALIDATION\033[0m")
    #Train and test on 6 different subsets of the data as we concatenated 6 datasets at the start
    #After splitting, cross validation will use each subset as testing-set once while training with other sets
    cv_scores = cross_val_score(classifier, x, y, cv=6)
    for i, subset_score in enumerate(cv_scores):
        print("Subset", i+1, "score:", subset_score)
    print("-> Mean accuracy:", cv_scores.mean())

if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "-s") or (len(sys.argv) > 2 and sys.argv[2] == "-s"):
        g_skip = False
    else:
        g_skip = True
    if (len(sys.argv) > 1 and sys.argv[1] == "-a") or (len(sys.argv) > 2 and sys.argv[2] == "-a"):
        algo = input("Select classifier algorithm (DecisionTree, KNN, GradientBoosting) : ")
    else:
        algo = "KNN"
    main(algo)

