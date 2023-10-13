import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from processing_EEGs_lib.preprocessing import preprocessing_transformation, get_data
from processing_EEGs_lib.dimensionality_reduction_algorithm import CSPTransformer

def print_shape(x, y):
    print("x:", x.shape)
    print("y:", y.shape)

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
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, 
                                                                shuffle=True)

print("\033[92mPROCESSING\033[0m")
pipe = Pipeline([
            ("dimensionality_reduction_algorithm", CSPTransformer(6))
        ])

x = pipe.fit_transform(x, y)
print_shape(x, y)

