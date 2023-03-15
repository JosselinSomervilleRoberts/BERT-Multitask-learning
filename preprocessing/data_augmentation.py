import pandas as pd
import sys
sys.path.append('./')

def generate_EDA_format(df, label_name, input_name, saving_path, classes_to_augment):
    """
    This function generates a dataframe with the format expected by EDA. (Easy Data Augmentation)
    It takes as input a dataframe df, a label name label_name and an input name input_name.
    It returns a dataframe with the format expected by EDA. (label /t input)
    """
    #We only keep the label and the input columns
    df_eda = df[[label_name, input_name]]
    #We want to augment the classes that are underrepresented
    df_eda = df_eda[df_eda[label_name].isin(classes_to_augment)]
    #We save to csv file
    df_eda.to_csv(save_path, sep='\t', index=False, header=False)

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
    df_merged.to_csv(saving_path, sep='\t', header=True)

if __name__ == "__main__":
    #We read the datasets
    df_train_sentiment = pd.read_csv('data/preprocessed_data/preprocessed-ids-sst-train.csv', sep='\t', header=0, index_col=0)
    save_path = 'data/preprocessed_data/EDA_data/input-EDA-format-ids-sst-train.txt'
    #We generate the EDA format needed for the data augmentation
    generate_EDA_format_flag = False
    if generate_EDA_format_flag:
        generate_EDA_format(df_train_sentiment, 'sentiment', 'sentence', save_path, [0,2,4])

    #We merge the augmented data generated using the code of the following repo (source : https://github.com/jasonwei20/eda_nlp)
    # to the original dataset
    augmented_data_path = 'data/preprocessed_data/EDA_data/output-EDA-format-ids-sst-train_augmented.txt'
    save_path = 'data/preprocessed_data/EDA_data/preprocessed-EDA-ids-sst-train.csv'
    merge_augmented_data(df_train_sentiment, augmented_data_path, 'sentiment', 'sentence', save_path, [0,2,4])


