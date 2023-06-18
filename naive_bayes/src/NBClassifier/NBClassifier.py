from numpy import mean,std
from math import sqrt,exp,pi
from src.preprocessing.preprocessing import separate_by_class

class NBClassifier:
    def __init__(self,data,dist):
        self.dist = dist
        self.data = data

    def get_mean(self,data_list):
        return mean(data_list)

    def get_std(self,data_list):
        return std(data_list)

    def get_summaries(self,df):
        summaries=[]
        #Select all columns except the last one
        filter_df = df.iloc[:,:-1]
        for col in filter_df.columns:
            #Summaries append format: [(avg,std,count),(avg,std,count),...]
            summaries = [(self.get_mean(filter_df[col]),self.get_std(filter_df[col]),len(filter_df[col])) for col in filter_df.columns]
        del summaries[-1]
        return summaries

    def summarize_by_class(self,df):
        separated = separate_by_class(df)
        summaries = {}
        for subdf in separated:
            class_name = subdf.iloc[:,-1].unique()[0]
            summaries[class_name] = self.get_summaries(subdf)
        return summaries

    def get_probability(self,x,mean,std):
        exponent = exp(-((x-mean)**2/(2*std**2)))
        output  = (1/(sqrt(2*pi)*std))*exponent
        return output

    def get_class_probability(self,summary,row,total_rows):
        probabilities = {}
        for class_value,class_summary in summary.items():
            probabilities[class_value] = summary[class_value][0][2]/total_rows
            for i in range(len(class_summary)):
                mean,std,count = class_summary[i]
                probabilities[class_value] *= self.get_probability(row[i],mean,std)
        return probabilities
    
    def predict(self, summaries, df):
        total_rows = len(df)
        for row in df.itertuples(index=False):
            probabilities = self.get_class_probability(summaries, row, total_rows)
            chosen_label, chosen_prob = max(probabilities.items(), key=lambda x: x[1])
            yield chosen_label


    def train(self,df_train):
        summary = self.summarize_by_class(df_train)
        return summary