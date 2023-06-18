def evaluate_prediction(class_value,predicted_class):
    if class_value == predicted_class:
        return 1
    else:
        return 0

def get_accuracy(df):
    correct = df['prediction_result'].sum()
    total = len(df)
    return correct/total