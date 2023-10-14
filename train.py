import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from processing_EEGs_lib.preprocessing import preprocessing_transformation, get_data
from processing_EEGs_lib.dimensionality_reduction_algorithm import CSPTransformer
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
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
    #x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, 
                                                                    #shuffle=True)
    return x_train, x_test, y_train, y_test

def processing(x, y):
    print("\033[92mPROCESSING\033[0m")
    steps = [
                #The recommended approach is to normalize the data after splitting it into training and testing sets. 
                #The rationale behind this recommendation is to prevent any information leakage from the testing set 
                #into the training set
                #If you need to perform feature importance (for example, for dimensionality reduction purposes), 
                #you must normalize your dataset in advance
                ("normalization", MinMaxScaler()), 
                #nb_components in CSP does not seem to have impact
                ("dimensionality-reduction-algorithm", CSPTransformer(nb_components=4))
            ]
    process_x_pipe = Pipeline(steps)
    print_pipe(process_x_pipe)
    x = process_x_pipe.fit_transform(x, y)
    print_shape(x, y)
    return steps

def train_test(steps, x_train, x_test, y_train, y_test):
    print("\033[92mTRAIN\033[0m")
    pipe = Pipeline(steps)
    print_pipe(pipe)
    pipe.fit(x_train, y_train)
    score_train = print_score(pipe, x_train, y_train, " training set:")
    print("\033[92mTEST\033[0m")
    score_test = print_score(pipe, x_test, y_test, " test set:")
    return score_train, score_test

def main():
    x, y = preprocessed_data()
    steps = processing(x, y)
    _continue()
    #For value k (n_neighbors) in practice values between 1-21 are used.
    #Higher values bias towards underfitting and vice-versa 
    #steps.append(("classifier", KNeighborsClassifier(n_neighbors=10))) #KNN did not work on test set nomatter the K value
    steps.append(("classifier", GradientBoostingClassifier(validation_fraction=0.2)))
    train_scores = []
    test_scores = []
    for _ in range(20):
        x_train, x_test, y_train, y_test = split_data(x, y)
        score_train, score_test = train_test(steps, x_train, x_test, y_train, y_test)
        _continue()
        train_scores.append(score_train)
        test_scores.append(score_test)
    print("\033[92mCONCLUSIONS\033[0m")
    print(" average score over 20 experiments of training set:", sum(train_scores)/len(train_scores))
    print(" average score over 20 experiments of test set:    ", sum(test_scores)/len(test_scores))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-s":
        g_skip = False
    else:
        g_skip = True
    main()

