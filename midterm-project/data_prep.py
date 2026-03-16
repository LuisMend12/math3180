import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
Load data → build dataframe → then:
X = df.drop(columns="spam")
y = df["spam"]

X_binary = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
   X_binary, y, test_size=0.3, random_state=42, stratify=y
)
