import pandas as pd
import os
import csv
import json
import re

def detect_encoding(file_path):
    import chardet
    with open(file_path, 'rb') as file:
        sample = file.read(10000)  # Read only a sample for efficiency
        result = chardet.detect(sample)
        return result['encoding']

def clean_data(file_info, output_file):
    """Clean data from specified files and save to a new CSV file, handling different encodings."""
    # Initialize the output file as blank
    open(output_file, 'w').close()

    fields = ['headline', 'sentiment']
    all_data = []

    for info in file_info:
        file_path = info[0]
        title_index = info[1]
        sentiment_index = info[2]
        encoding = detect_encoding(file_path)
        global test
        global t
        
        with open(file_path, mode='r', encoding=encoding)as file:
            if file_path.endswith('.csv'):
                csvFile = csv.reader(file)

                for line in csvFile:
                    s = convert_sentiment(line[sentiment_index])
                    h = re.sub(r'[^A-Za-z0-9%., ]+', '', line[title_index])
                    h = h.replace('%', 'percent')
                    if type(s) == int:
                        all_data.append([h, s])
            
            elif file_path.endswith('.txt'):
                txtFile = file.readlines()

                for line in txtFile:
                    l = line.split('@')
                    
                    s = convert_sentiment(l[sentiment_index])
                    h = re.sub(r'[^A-Za-z0-9%., ]+', '', l[title_index])
                    h = h.replace('%', 'percent')
                    if type(s) == int:
                        all_data.append([h, s])
    
    df = pd.DataFrame(all_data, columns=fields)
    df.to_csv(output_file, index=False)
    
    return output_file

def convert_sentiment(value):
    """Convert sentiment values to numeric format."""
    if value.strip().lower() == 'positive' or value.strip() == '1':
        return 1
    elif value.strip().lower() == 'negative' or value.strip() == '0':
        return 0
    else:
        try:
            temp = json.loads(value)
            pos = 0
            neg = 0

            for key in temp:
                if temp[key].strip().lower() == 'positive':
                    pos += 1
                elif temp[key].strip().lower() == 'negative':
                    neg += 1
                
            if pos - neg > 0:
                return 1
            elif pos-neg < 0:
                return 0
        except:
            pass
    
    return None

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    file_info = [
        [f'{dir_path}\\raw_data\\all-data.csv', 1, 0], 
        [f'{dir_path}\\raw_data\\data.csv', 0, 1], 
        [f'{dir_path}\\raw_data\\Fin_Cleaned.csv', 1, 4],
        [f'{dir_path}\\raw_data\\Sentences_75Agree.txt', 0, 1],
        [f'{dir_path}\\raw_data\\SEntFiN-v1.1.csv', 1, 2],
    ]

    output_file = f'{dir_path}\\cleaned_data.csv'
    clean_data(file_info, output_file)