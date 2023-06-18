

def separate_by_class(df) -> list:
    """Splits a dataframe into smaller datasets by class

    Args:
        df ([DataFrame]): dataframe to be separated

    Returns:
        list: [DataFrame Class1, DataFrame Class2,...] Array of dataframes separated by each class
    """    
    
    output = []
    unique_classes = df.iloc[:,-1].unique()
    for unique_class in unique_classes:
        output.append(df[df.iloc[:,-1] == unique_class])
    return output


def split_dataset(df, percentage):
    """Splits a dataset into train and test sets.
    Args:
        df (DataFrame): The dataset to split.
        percentage ([int]): The percentage of the dataset to use for training.
    Returns:
        [set]: Returns the train and test sets.
    """    
    # Sample creates a shuffle of the dataset by default.
    df_train = df.sample(frac = percentage)
    #To get the test set we need to remove the rows from the training set
    df_test = df.drop(df_train.index)
    return (df_train, df_test)