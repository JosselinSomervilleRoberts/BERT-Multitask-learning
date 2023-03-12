'''
This file analyzes the datasets that we are using.
'''
from datasets import *
import pandas as pd

def add_length_column(df, task):
    '''
    This function adds a column to the dataset with the length of the input sentence/sentences.
    '''
    if (task == "sentiment"):
        df["length"] = df["sentence"].apply(lambda x: len(x))
    else : 
        #We add the length of the two sentences
        df["length"] = df["sentence1"].apply(lambda x: len(x)) + df["sentence2"].apply(lambda x: len(x))

def print_length_statistics(df, task):
    '''
    This function computes the average length of the sentences in the dataset and other statistics.
    Each element in the dataset is of the form (sentence, label, ID). It prints the results and returns None.
    '''
    #We add a length column
    add_length_column(df, task)
    print(task + " :\n")
    print(df["length"].describe())
    print('\n')

def print_class_repartition(df, task):
    """
    This function computes the repartition of the classes in the dataset and prints it.
    """
    # Compute the repartition of the classes
    if (task == "similarity"):
        raise ValueError("The similarity task does not have classes.")
    else:
        label = task
        repartition = df[label].value_counts()
        print(repartition)

def generate_preprocessed_csv(df, task, df_path, max_length = 210, saving_path = 'data/preprocessed_data/'):
    """
    This function generates a csv file with the preprocessed dataset with respect to lengths.
    It takes as input a dataframe df, a task, a dataframe path df_path, a maximimum length max_length
    and a saving path saving_path.
    
    The dataset is of the form (sentence, label, ID) for the sentiment task and 
    (sentence1, sentence2, label) for the paraphrase task.
    """
    #We add a length column
    add_length_column(df, task)

    #We only need the ids of the rows that we keep
    new_df = df[df["length"] <= max_length]

    #We read the initial dataframe
    initial_df = pd.read_csv(df_path, sep='\t', header=0, index_col=0)

    #We filter the initial dataframe with the ids of the rows that we keep
    initial_length = len(initial_df)
    final_df = initial_df[initial_df.id.isin(new_df.id)]
    final_length = len(final_df)
    difference = initial_length - final_length
    print("We drop {} training examples ({}%).".format(difference,round(difference/initial_length*100, ndigits=2)))

    # Save the dataset
    filename = df_path.split("/")[-1]
    final_df.to_csv(saving_path + "preprocessed-" + filename, index=True, sep='\t', header=True)

    
            
if __name__ == '__main__' :
    # Load the datasets
    sentiment_filename = 'data/ids-sst-train.csv'
    paraphrase_filename = 'data/quora-train.csv'
    similarity_filename = 'data/sts-train.csv'
    sentiment_data, num_labels, paraphrase_data, similarity_data = load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename)

    # Create dataframes
    df_sentiment = pd.DataFrame(sentiment_data, columns=['sentence', 'sentiment', 'id'])
    df_paraphrase = pd.DataFrame(paraphrase_data, columns=['sentence1', 'sentence2', 'paraphrase', 'id'])
    df_similarity = pd.DataFrame(similarity_data, columns=['sentence1', 'sentence2', 'similarity', 'id'])

    # Analyze the datasets
    #Print the length statistics and the class repartition
    print("Sentiment dataset :\n")
    print_length_statistics(df_sentiment, "sentiment")
    print_class_repartition(df_sentiment, "sentiment")
    generate_preprocessed_csv(df_sentiment, "sentiment", sentiment_filename, max_length = 300)
    print("\n")

    print("Paraphrase dataset :\n")
    print_length_statistics(df_paraphrase, "paraphrase")
    print_class_repartition(df_paraphrase, "paraphrase")
    generate_preprocessed_csv(df_paraphrase, "paraphrase", paraphrase_filename, max_length = 300)
    print("\n")

    print("Similarity dataset :\n")
    print_length_statistics(df_similarity, "similarity")
    generate_preprocessed_csv(df_similarity, "similarity", similarity_filename, max_length = 300)
    print("\n")

