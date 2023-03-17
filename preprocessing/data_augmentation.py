import pandas as pd
import sys
sys.path.append('./')

def generate_EDA_format(df, label_name, input_name, saving_path):
    """
    This function generates a dataframe with the format expected by EDA. (Easy Data Augmentation)
    It takes as input a dataframe df, a label name label_name and an input name input_name.
    It returns a dataframe with the format expected by EDA. (label /t input)
    """
    #We only keep the label and the input columns
    df_eda = df[[label_name] + input_name]
    #We get the counts of each label in the dataset
    counts = df_eda[label_name].value_counts()
    print(counts)
    most_common_label = counts.idxmax()
    possible_labels = df_eda[label_name].unique()
    class_to_augment = dict()
    for label in possible_labels:
        class_to_augment[label] = counts[most_common_label] - counts[label]
        print(class_to_augment[label])
    #We want to augment the classes that are underrepresented
    #We shuffle once the dataframe
    df_eda = df_eda.sample(frac=1).reset_index(drop=True)
    
    if (label_name == "sentiment"):
        #We select the n//2 random training examples of each underepresented class since we will augment them by one
        #using the EDA algorithm (Easy Data Augmentation)
        nb_0 = class_to_augment[0]//2
        class_0 = df_eda[df_eda["sentiment"] == 0].head(nb_0)
        nb_1 = class_to_augment[1]//2
        class_1 = df_eda[df_eda["sentiment"] == 1].head(nb_1)
        nb_2 = class_to_augment[2]//2
        class_2 = df_eda[df_eda["sentiment"] == 2].head(nb_2)
        nb_4 = class_to_augment[4]//2
        class_4 = df_eda[df_eda["sentiment"] == 4].head(nb_4)
        final_df_eda_format = pd.concat([class_0, class_1, class_2, class_4])
        final_df_eda_format.reset_index()
    if (label_name == "is_duplicate"):
        nb_1 = class_to_augment[1]//2
        class_1 = df_eda[df_eda["is_duplicate"] == 0.0].head(nb_1)
        final_df_eda_format = class_1
        final_df_eda_format.reset_index()
    if (label_name == "similarity"):
        nb_1 = class_to_augment[1]//2
        class_1 = df_eda[df_eda["similarity"] == 1.0].head(nb_1)
        final_df_eda_format = class_1
        final_df_eda_format.reset_index()
    #We save to csv file
    final_df_eda_format.to_csv(saving_path, sep='\t', index=False, header=False)

def merge_augmented_data(df, augmented_data_path, label_name, input_name, saving_path, classes_to_augment):
    """
    This function merges the augmented data to the original dataset."""
    #We drop the underrepresented classes
    print(len(df))
    df = df[~df[label_name].isin(classes_to_augment)]
    print(len(df))
    #We read the augmented data
    df_augmented = pd.read_csv(augmented_data_path, sep='\t', header=None, index_col=None)
    #We rename the columns
    df_augmented.rename(columns={0:label_name, 1:input_name}, inplace=True)
    print(len(df_augmented))
    #We merge the dataframes
    df_merged = pd.concat([df, df_augmented], ignore_index=True)
    #df_merged.to_csv(saving_path, sep='\t', header=True)

if __name__ == "__main__":
    #We read the datasets
    df_train_sentiment = pd.read_csv('data/preprocessed_data/lengths/preprocessed-ids-sst-train.csv', sep='\t', header=0, index_col=0)
    save_path_sentiment = 'data/preprocessed_data/EDA_data/input-EDA-format-ids-sst-train.txt'

    df_train_paraphrase = pd.read_csv('data/preprocessed_data/lengths/preprocessed-quora-train.csv', sep='\t', header=0, index_col=0)
    save_path_paraphrase = 'data/preprocessed_data/EDA_data/input-EDA-format-quora-train.txt'

    #We generate the EDA format needed for the data augmentation
    generate_EDA_format_flag = False
    if generate_EDA_format_flag:
        generate_EDA_format(df_train_sentiment, 'sentiment', ['sentence'], save_path_sentiment)
        #generate_EDA_format(df_train_paraphrase, 'is_duplicate', ['sentence1', 'sentence2'], save_path_paraphrase)

    #We merge the augmented data generated using the code of the following repo (source : https://github.com/jasonwei20/eda_nlp)
    # to the original dataset
    augmented_data_path = 'data/preprocessed_data/EDA_data/output-EDA-format-ids-sst-train_augmented.txt'
    save_path = 'data/preprocessed_data/EDA_data/preprocessed-EDA-ids-sst-train.csv'
    merge_augmented_data(df_train_sentiment, augmented_data_path, 'sentiment', 'sentence', save_path, [0,2,4])


