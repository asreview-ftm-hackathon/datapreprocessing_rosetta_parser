from pathlib import Path
from string import punctuation
from collections import Counter, defaultdict
from heapq import nlargest
import re

import spacy
from spacy.lang.nl.stop_words import STOP_WORDS
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# Paths, constants and models
input_data_folder = Path('./input_data')
output_file = Path('./output_data') / 'processed_data.csv'
spacy_model = spacy.load('nl_core_news_lg')
N_SUMMARY_SENTENCES = 5

# Helper for development, quick sanity checks
def check_column(df_series: pd.Series,
                 for_nans: bool = True,
                 for_duplicates: bool = True,
                 for_leading_spaces: bool = True,
                 for_trailing_spaces: bool = True,
                 ):
    if for_nans and len(df_series[df_series.isna()]):
        print('Found NaNs. Not checking further.')
        return
    if for_duplicates and len(df_series[df_series.duplicated(keep=False)]):
        print('Found duplicates')
    if for_leading_spaces and len(df_series[df_series.str.startswith(' ')]):
        print('Found leading spaces')
    if for_trailing_spaces and len(df_series[df_series.str.endswith(' ')]):
        print('Found trailing spaces')

# Text normalization, will be useful later
def normalize_text(string):
    """ copy-paste from: https://github.com/koreyou/text-gcn-chainer/nlp_utils.py"""
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Find all .csv files in input_data folder and load them into a dataframe
input_files = [str(f) for f in input_data_folder.rglob('*.csv')]
input_dfs = [pd.read_csv(f, sep='|') for f in input_files]
df = pd.concat(input_dfs)

# Rename columns; clean noisy rows; remove system specific files; reset index but save original index too
df.columns = ['filename', 'file_content']
df = df.dropna().drop_duplicates()
df = df[~df.filename.isin(['.DS_Store', 'nan', 'Readme.md'])]
df = df.reset_index(drop=True)

# Extract ID and strip leading/trailing periods
df['id'] = df['filename'].str.extract('(?P<id>[0-9\.]+)', expand=True)
df['id'] = df['id'].str.lstrip('.')  # btw lstrip can accept more than one character
check_column(df['id'])

# Extract category, replace missing with "Onbekend", strip and capitalize
df['category'] = df['filename'].\
    str.extract('(?P<category>(?<=[0-9\_]\_)[a-z A-Z]+(?=\_))', expand=True)
df['category'] = df['category'].fillna('Onbekend')
df['category'] = df['category'].str.lstrip().str.capitalize()
check_column(df['category'], for_duplicates=False)

# Extract date from filename and replace missing dates with ?
df['filename_date'] = df['filename'].str.extract('(?P<filename_date>[0-9-?]+(?=.pdf$))', expand=True)
df['filename_date'] = df['filename_date'].fillna('?')
df['filename_date'] = df['filename_date'].str.replace('^[^0-9]+', '?', regex=True)
check_column(df['filename_date'], for_duplicates=False)

# Try to get date from data contents
df['parsed_date'] = df['file_content'].str.extract('(?P<parsed_date>[0-9-]{3,5}-20[0-9]{2} [0-9:]{8})')
df['parsed_date'] = df['parsed_date'].fillna('?')
check_column(df['filename_date'], for_duplicates=False)

# Try to parse stuff in contents with beautiful soup and spacy
df['found_emails'] = '?'
df['found_urls'] = '?'
df['found_organizations'] = '?'
df['found_people'] = '?'
df['found_dates'] = '?'

df['summary'] = '?'
df['top5words'] = '?'
df['title'] = '?'
df['abstract'] = '?'

for i in tqdm(range(len(df['file_content'])), desc='Parsing documents'):
    soup = BeautifulSoup(df['file_content'][i], 'html.parser')
    contents = soup.get_text()
    tokens = spacy_model(contents)

    # Emails
    found_unique_emails = set([d.text.strip() for d in tokens if d.like_email])
    df.at[i, 'found_emails'] = list(found_unique_emails)

    # URLs
    found_unique_urls = set([d.text.strip() for d in tokens if d.like_url])
    df.at[i, 'found_urls'] = list(found_unique_urls)

    # Named entities: organizations, people and dates
    found_unique_organizations = set([f.text.strip() for f in tokens.ents if f.label_ == 'ORG'])
    df.at[i, 'found_organizations'] = list(found_unique_organizations)
    found_unique_people = set([f.text.strip() for f in tokens.ents if f.label_ == 'PERSON'])
    df.at[i, 'found_people'] = list(found_unique_people)
    found_unique_dates = set([f.text.strip() for f in tokens.ents if f.label_ == 'DATE'])
    df.at[i, 'found_dates'] = list(found_unique_dates)

    # Summarization and top5 keywords
    ## Find all keywords in the contents
    keywords = []
    for token in tokens:
        if (token.text in list(STOP_WORDS)) or (token.text in punctuation):
            continue
        if token.pos_ in ['PROPN', 'ADJ', 'NOUN', 'VERB']:
            keywords.append(token.text)
    ## Count word frequency and normalize it
    freq_words = Counter(keywords)
    df.at[i, 'top5words'] = [w[0] for w in freq_words.most_common(5)]
    max_freq = max(freq_words.values())
    for word in freq_words.keys():
        freq_words[word] = freq_words[word] / max_freq
    ## Sentence weighting
    sent_strength = defaultdict(int)
    for sent in tokens.sents:
        for word in sent:
            if word.text in freq_words.keys():
                sent_strength[sent] += freq_words[word.text]
    ## Finding N_SUMMARY_SENTENCES most important sentences
    heaviest_sentences = nlargest(N_SUMMARY_SENTENCES, sent_strength, key=sent_strength.get)
    summary = ' '.join([w.text.replace('\n', '') for w in heaviest_sentences])
    df.at[i, 'summary'] = summary

    # Cleaned title and contents
    df.at[i, 'title'] = normalize_text(df['filename'][i])
    df.at[i, 'abstract'] = contents.replace('\n', '')

# Write down to csv file
print(f'Saving data to {str(output_file)}...', end='')
output_file.parent.mkdir(exist_ok=True)
df.to_csv(output_file, sep='|')
print(f' Done!')
