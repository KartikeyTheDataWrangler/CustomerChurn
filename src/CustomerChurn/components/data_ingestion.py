import os
import sys
from src.CustomerChurn.exception import CustomException
from src.CustomerChurn.exception import logging
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
