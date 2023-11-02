import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.CustomerChurn.logger import logging
from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.utils import save_object
from src.CustomerChurn.utils import Col_Dropper

from sklearn.pipeline import Pipeline
 # Replace 'your_module' with the actual module where your custom transformer is defined

# Create a pandas DataFrame (assuming 'df' is your DataFrame)
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print(df)

# Create the pipeline with your custom transformer
pipeline = Pipeline(
    [
        ('coldrop',Col_Dropper(coltodrop=['B']))
    ]
)

# Fit and transform the DataFrame using the pipeline
transformed_df = pipeline.fit_transform(df)

print(transformed_df)
