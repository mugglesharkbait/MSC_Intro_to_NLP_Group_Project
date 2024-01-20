import pandas as pd
from nltk import ngrams
from collections import defaultdict

def read_data():
    df_list=[]
    for n in range(1,6):
        df = pd.read_csv(f'./output/{n}-gram.txt', sep='\t')
        df_list.append(df)
    return df_list
def evaluate_sentence(sentence, ngram_data_list):
    average_scores=[]
    # Divide the sentence into n-grams
    for n in range(1, 6):
        # Initialize total score and count for averaging
        total_score = 0
        ngrams_list = list(ngrams(sentence.split(), n))
        count = 0
        for ngram in ngrams_list:
            ngram_key = ' '.join(ngram)
            # Check if the n-gram entry exists in the aggregated data
            if ngram_key in ngram_data_list[n-1]['n-gram'].values:
                # Get the total score and total occurrences from the aggregated data
                ngram_entry = ngram_data_list[n-1][ngram_data_list[n-1]['n-gram'] == ngram_key]
                total_score += ngram_entry['average_score'].values[0]
                count += 1
        # Calculate the average score for the sentence
        average_score = total_score / count if count > 0 else 0
        average_scores.append(average_score)
    return average_scores


# Example usage
sentence = "Mr. Ramesh said 70 percent of India's iron ore lay in states infiltrated by Maoists; production in this area is stalled at 16 million tons a year even though the area has the potential to produce 100 million tons."
result = evaluate_sentence(sentence, read_data())
print(f"Sentence: `{sentence}`\nScore: {result}")

