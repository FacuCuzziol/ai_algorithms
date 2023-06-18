import streamlit as st
import pandas as pd
import numpy as np
import time
from src.NBClassifier.NBClassifier import NBClassifier
from src.metrics.classification_metrics import evaluate_prediction,get_accuracy
from src.preprocessing.preprocessing import split_dataset

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except:
        st.error("Error loading the dataset. Please make sure the file is a properly formatted CSV.")
        
def train_model(df, split_ratio):
    df_train, df_test = split_dataset(df, split_ratio)

    classifier = NBClassifier(df_train, 'gaussian')
    summary = classifier.train(df_train)
    st.write('Train dataset size:', len(df_train))
    st.write('Test dataset size:', len(df_test))

    return classifier, summary, df_test

def evaluate_model(df_test, classifier):
    df_test['predicted_class'] = list(classifier.predict(summary, df_test))
    df_test['prediction_result'] = df_test.apply(lambda row: evaluate_prediction(row['class'], row['predicted_class']), axis=1)
    accuracy = get_accuracy(df_test)

    return df_test, accuracy

#if main
if __name__ == '__main__':
    # Main Streamlit app
    st.title("Naive Bayes Classifier")
    st.write("Step 1: Dataset upload")
    # Step 1: Upload file
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if file is not None:
        #Load the dataset
        df = load_dataset(file)
        
        st.write('Step 2: Split ratio')
        #Step 2: specify split ratio
        split_ratio = st.number_input("Enter the split ratio for train/test", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Step 3: Train the model
        if st.button("Train"):
            try:
                classifier, summary, df_test = train_model(df, split_ratio)
                st.write("Training complete!")

                # Step 4: Get results
                st.write("Results:")
                df_results, accuracy = evaluate_model(df_test, classifier)
                st.write(df_results)

                # Step 5: Show accuracy
                st.write("Accuracy:", round(accuracy,2))
            except Exception as e:
                st.error(f'Error training the model: {e}')
