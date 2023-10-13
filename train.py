import numpy as np
from sklearn.pipeline import Pipeline
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

print("\033[92mPROCESSING\033[0m")
pipe = Pipeline([
            ("dimensionality_reduction_algorithm", CSPTransformer(y, 6))
        ])

x = pipe.fit_transform(x)
print_shape(x, y)

