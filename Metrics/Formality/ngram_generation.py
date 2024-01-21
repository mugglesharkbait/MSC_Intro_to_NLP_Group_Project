import pandas as pd
from nltk import ngrams
from collections import defaultdict
import chardet

# Having tab-separated files
answers = 'annotated_sentences/answers.tsv'
blog = 'annotated_sentences/blog.tsv'
email = 'annotated_sentences/email.tsv'
news = 'annotated_sentences/news.tsv'
file_paths = [answers, blog, email, news]
encodings = []
for file_path in file_paths:
    with open(file_path, 'rb') as file:
        encodings.append(chardet.detect(file.read())['encoding'])


def read_data():
    dfs = []
    i = 0
    for file_path in file_paths:
        # Read the tab-separated data using pandas
        df = pd.read_csv(file_path, sep='\t', encoding=encodings[i],header=None)
        i += 1
        # Append the DataFrame to the list
        dfs.append(df)
    # Combine all DataFrames into a single DataFrame
    return pd.concat(dfs)


def process(df):
    # Create dictionaries to store n-gram data
    ngram_data = {1: defaultdict(lambda: [0, 0,0]),
                  2: defaultdict(lambda: [0, 0,0]),
                  3: defaultdict(lambda: [0, 0,0]),
                  4: defaultdict(lambda: [0, 0,0]),
                  5: defaultdict(lambda: [0, 0,0])}

    # Process each row efficiently using vectorized operations
    for index, row in df.iterrows():
        score = float(row[0])
        sentence = row[3]

        for n in range(1, 6):
            ngrams_list = list(ngrams(sentence.split(), n))
            for ngram in ngrams_list:
                ngram_key = ' '.join(ngram)
                ngram_data[n][ngram_key][0] += score
                ngram_data[n][ngram_key][1] += 1
                ngram_data[n][ngram_key][2] = ngram_data[n][ngram_key][0]/ngram_data[n][ngram_key][1]

    # Write n-gram data to separate txt files
    for n in range(1, 6):
        output_file = f'./output/{n}-gram.txt'
        ngram_df = pd.DataFrame.from_dict(ngram_data[n], orient='index', columns=['total_score', 'total_occurrences','average_score'])
        ngram_df.index.name = 'n-gram'
        ngram_df.reset_index(inplace=True)
        ngram_df.to_csv(output_file, sep='\t', index=False)
        print(f"Data successfully processed and written to files in {output_file}")


df = read_data()
process(df)
