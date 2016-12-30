import pandas as pd
from bs4 import BeautifulSoup
from StopWords import stop_words
from collections import defaultdict
import math
import csv


debug = False
poll = 1000
limit = 1000

def clean_html(data):
    print('> cleaning html tags....')
    for i in range(len(data)):
        data[i] = BeautifulSoup(data[i],'html.parser').get_text().replace('\n',' ')
        if debug:
            if not i % poll:
                print('> clean_html proceeding: ' + str(i) + '/' + str(len(data)))

    return data

def delete_stopwords(data):
    return [word for word in data if word not in stop_words]

def setup_text(data):
    print('> setting up text....')
    data_list = list()
    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    for i, row in enumerate(data):
        data_list.append(delete_stopwords([word.strip().lower() for word in word_split.split(text)]))
        if debug:
            if not i % poll:
                print('> setup_text proceeding: ' + str(i) + '/' + str(len(data)))

    return data_list

def build_tf_table(data):
    print('> building tf_table....')
    tf_table = list()
    for i, row in enumerate(data):
        word_count = 0
        tfFreqDict = defaultdict(int)
        for word in row:
            tfFreqDict[word] += 1
            word_count += 1
        for word in tfFreqDict:
            tfFreqDict[word] = tfFreqDict[word] / word_count

        tf_table.append(tfFreqDict)
        if debug:
            if not i % poll:
                print('> tf_table proceeding: ' + str(i) + '/' + str(len(data)))

    return tf_table

def build_idf_table(word_list, data):
    print('> building idf_table....')
    idf_table = defaultdict(int)
    for word in word_list:
        idf_table[word] = 0

    for i, row in enumerate(data):
        for word in set(row):
            idf_table[word] += 1
        if debug:
            if not i % poll:
                print('> idf_table proceeding: ' + str(i) + '/' + str(len(data)))

    for word in idf_table:
        idf_table[word] = math.log((len(data)+1)/(idf_table[word]))

    return idf_table

def create_word_list(data):
    print('> creating word list....')
    return map(str, sum(data, []))

def build_tfidf_table(data, value_k = 4):
    # Ranking Algorithm uses BM25 Algorithm which considers not only tfidf value
    # but also length of content(using pivoting) and reducing gaps of score by
    # word frequency.

    print('> starting tfidf_table....')
    tf_table = build_tf_table(data)
    idf_table = build_idf_table(create_word_list(data), data)

    print('> building tfidf_table....')
    tfidf_table = list()
    for i, row in enumerate(tf_table):
        tfidfDict = defaultdict(int)
        for word in row:
            tfidfDict[word] = (((value_k+1)*row[word])/(row[word]+value_k))*idf_table[word]

        tfidf_table.append(tfidfDict)
        if debug:
            if not i % poll:
                print('> tfidf_table proceeding: ' + str(i) + '/' + str(len(tf_table)))

    return tfidf_table

def set_tags(data, length=3):
    tag_list = list()
    for row in data:
        tag = set(sorted(row, key = row.get, reverse = True)[:length])
        tag_list.append(' '.join(tag))

    return tag_list

def save_csv(data, file_name, path = ""):
    with open(path + file_name + '.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'tags'])
        for i in range(len(data)):
            csv_writer.writerow([data['id'][i], data['tags'][i]])


test_data = pd.read_csv('./src/test.csv')
test_data_content_np = test_data['content'].as_matrix()
content_data = setup_text(clean_html(test_data_content_np))
tag_list = set_tags(build_tfidf_table(content_data))
result = pd.DataFrame({'id':test_data['id'], 'tags':tag_list})
save_csv(result, 'test')
