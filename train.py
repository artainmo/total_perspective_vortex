import numpy as np
from sklearn.pipeline import Pipeline
from processing_EEGs_lib.preprocessing import preprocessing_transformation, get_data
from processing_EEGs_lib.dimensionality_reduction_algorithm import CSPTransformer

raw_data, annotations = get_data()
x, y = preprocessing_transformation(raw_data, annotations)

pipe = Pipeline([
            ("dimensionality_reduction_algorithm", CSPTransformer())
        ])

processed_x = pipe.fit_transform(x)
print(processed_x.shape, y.shape)
print(np.unique(y))

