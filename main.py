import pandas as pd
import numpy as np

from src.NBClassifier.NBClassifier import NBClassifier
from src.metrics.classification_metrics import evaluate_prediction,get_accuracy
from src.preprocessing.preprocessing import split_dataset


#if main
if __name__ == '__main__':
    df = pd.read_csv('data/iris.csv')
    NBClassifier = NBClassifier(df,'gaussian')
    df_train,df_test = split_dataset(df,0.7)

    summary = NBClassifier.train(df_train)
    print(f'SUMMARY: {summary}')

    df_test['predicted_class'] = NBClassifier.predict(summary,df_test)

    df_test['prediction_result'] = df_test.apply(lambda row: evaluate_prediction(row['class'],row['predicted_class']),axis=1)

    print(f'ACCURACY: {get_accuracy(df_test)}')

