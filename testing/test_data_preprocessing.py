import sys
sys.path.append('src')
import unittest
import pandas as pd
from datasets_preprocessing_analysis import *

class TestStringMethods(unittest.TestCase):

    def test_add_length(self):
        # Load the datasets
        sentiment_filename = 'test_data/sst-train-initial-example.csv'
        paraphrase_filename = 'test_data/quora-train-initial-example.csv'
        similarity_filename = 'test_data/sts-train-initial-example.csv'
        sentiment_data, _, paraphrase_data, similarity_data = load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename)
        # Create dataframes
        df_sentiment = pd.DataFrame(sentiment_data, columns=['sentence', 'sentiment', 'id'])
        df_paraphrase = pd.DataFrame(paraphrase_data, columns=['sentence1', 'sentence2', 'paraphrase', 'id'])
        df_similarity = pd.DataFrame(similarity_data, columns=['sentence1', 'sentence2', 'similarity', 'id'])
        # Add a column with the length of the sentences
        add_length_column(df_sentiment, "sentiment")
        add_length_column(df_paraphrase, "paraphrase")
        add_length_column(df_similarity, "similarity")

        # Check the results
        self.assertEqual(df_sentiment['length'][0], 181)
        self.assertEqual(df_paraphrase['length'][0], 79)
        self.assertEqual(df_similarity['length'][0], 72)
        


    def test_generate_csv(self):
        # Load the datasets
        sentiment_filename = 'test_data/sst-train-initial-example.csv'
        paraphrase_filename = 'test_data/quora-train-initial-example.csv'
        similarity_filename = 'test_data/sts-train-initial-example.csv'
        sentiment_data, _, paraphrase_data, similarity_data = load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename)
        # Create dataframes
        df_sentiment = pd.DataFrame(sentiment_data, columns=['sentence', 'sentiment', 'id'])
        df_paraphrase = pd.DataFrame(paraphrase_data, columns=['sentence1', 'sentence2', 'paraphrase', 'id'])
        df_similarity = pd.DataFrame(similarity_data, columns=['sentence1', 'sentence2', 'similarity', 'id'])

        # Generate the csv
        generate_preprocessed_csv(df_sentiment, "sentiment", sentiment_filename, max_length = 210, saving_path='test_data/preprocessed_data/')
        generate_preprocessed_csv(df_paraphrase, "paraphrase", paraphrase_filename, max_length = 110, saving_path='test_data/preprocessed_data/')
        generate_preprocessed_csv(df_similarity, "similarity", similarity_filename, max_length = 200, saving_path='test_data/preprocessed_data/')
        
        # Check the results
        df_sentiment_preprocessed = pd.read_csv('test_data/preprocessed_data/preprocessed-sst-train-initial-example.csv', sep='\t', header=0, index_col=0)
        df_paraphrase_preprocessed = pd.read_csv('test_data/preprocessed_data/preprocessed-quora-train-initial-example.csv', sep='\t', header=0, index_col=0)
        df_similarity_preprocessed = pd.read_csv('test_data/preprocessed_data/preprocessed-sts-train-initial-example.csv', sep='\t', header=0, index_col=0)


        self.assertEqual(len(df_sentiment_preprocessed), 4)
        self.assertEqual(len(df_paraphrase_preprocessed), 4)
        self.assertEqual(len(df_similarity_preprocessed), 4)

if __name__ == '__main__':
    unittest.main()