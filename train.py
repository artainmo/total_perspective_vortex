import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from processing_EEGs_lib.preprocessing import preprocessing_transformation, get_data
from processing_EEGs_lib.dimensionality_reduction_algorithm import CSPTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from random import randint

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
    print("", pipe.score(x,y))

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

# Split in 0.6 (training), 0.2 (test), 0.2 (validation) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)
#x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, 
                                                                #shuffle=True) #validation set not usable in KNN

print("\033[92mPROCESSING\033[0m")
steps = [
            ("normalization", MinMaxScaler()), 
            #The recommended approach is to normalize the data after splitting it into training and testing sets. 
            #The rationale behind this recommendation is to prevent any information leakage from the testing set 
            #into the training set
            #If you need to perform feature importance (for example, for dimensionality reduction purposes), 
            #you must normalize your dataset in advance
            ("dimensionality-reduction-algorithm", CSPTransformer(4))
        ]
 
process_x_pipe = Pipeline(steps)
print_pipe(process_x_pipe)
x = process_x_pipe.fit_transform(x, y)
print_shape(x, y)

print("\033[92mTRAIN\033[0m")
steps.append(("classifier", KNeighborsClassifier(n_neighbors=8)))
pipe = Pipeline(steps)
print_pipe(pipe)
pipe.fit(x_train, y_train)
print_score(pipe, x_train, y_train, " training set:")

print("\033[92mTEST\033[0m")
print_score(pipe, x_test, y_test, " test set:")
