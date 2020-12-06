# import warnings
# import seaborn as sns
# import matplotlib.pylab as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import classification_report
import sys

# nltk.download('stopwords')
# warnings.simplefilter("ignore")
# plt.style.use('ggplot')
# color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# inline emoticons regex
emoticons_str = r"""
        (?:
        [:=;] #eye
        [oO\-]? # nose
        [D\)\]\(\]/\\OpP] # mouth
        )"""

# deep cleaner regex
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @someone
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # Hashtag
    # URLs
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',
    r"(?:[a-z][a-z'\-_]+[a-z])",  # Words containing - and ‘
    r'(?:[\w_]+)',  # others
    r'(?:\S)'  # others
]

# tokenize tweet for cleaning
tokens_re = re.compile(r'('+'|'.join(regex_str)+')',
                       re.VERBOSE | re.IGNORECASE)
# emoticon removal using regex
emoticon_re = re.compile(r'^'+emoticons_str+'$',
                         re.VERBOSE | re.IGNORECASE)

# deep cleaing which removes inline emoticons, html tags, urls, tagged people etc
# returns junk free tokens


def inline_cleaning(s, lowercase=False):
    tokens = tokens_re.findall(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in
                  tokens]
    return tokens

# removes URL from the tweet using regex
# returns a string without any url


def remove_urls(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                  '', text, flags=re.MULTILINE)
    return(text)

# removes everything expect # , @, _, $ or % and numbers and characters from the tweets using regex
# returns a string without punctuation


def remove_punctuation(word):
    # , @, _, $ or %
    rule = re.compile(r"[^a-zA-Z0-9\#\@\_\$\%]")
    word = rule.sub('', word)
    return word

# removes blank words or words which has length less than 2
# returns an array of strings without "junk"


def remove_junk(word_list):
    res = []
    for word in word_list:
        word = remove_punctuation(word)
        if word == '':
            continue
        res.append(word)
    return res

# facilitates in preprocessing process
# returns junk free dataframe


def preprocess(data, model):
    # convert every single word into lowercase
    # data['tweet_text'] = data['tweet_text'].str.lower()
    
    # calling remove_urls() for each tweet
    data['tweet_text'] = data['tweet_text'].apply(lambda x: remove_urls(x))

    # calling inline_cleaning() for each tweet
    data['words'] = data['tweet_text'].apply(lambda x: inline_cleaning(x))
    
    # calling remove_junk() for word
    data['words'] = data['words'].apply(lambda x: remove_junk(x))

    # remove stopwords and stemming for mnb only
    if model == 'mnb':
        # remove stopwords from the vocabulary
        stop_words = stopwords.words('english')
        data['words'] = data['words'].apply(
            lambda x: [i for i in x if i not in stop_words])

        # stem each word from the vocabulary
        stemmer = nltk.stem.PorterStemmer()
        data['words'] = data['words'].apply(
            lambda x: [stemmer.stem(i) for i in x])

    return data


# main function
if __name__ == '__main__':
    # command line arguments for file input
    dataset_path = sys.argv[1]
    testset_path = sys.argv[2]
    # training dataframe
    train = pd.read_csv(dataset_path, sep='\t', header=None, names=[
        'instance_number', 'tweet_text', 'sentiment'])
    # test dataframe    
    test = pd.read_csv(testset_path, sep='\t', header=None, names=[
        'instance_number', 'tweet_text', 'sentiment'])
    # merging the dataframes
    data_all = pd.concat([train, test], axis=0)
    # ground truth
    y_label = np.array(test['sentiment'])

    # print("--------------------------dt")
    # preproccesing the whole dataframe for dt
    data_dt = preprocess(data_all, 'dt')

    # clean_tweet_text column which contains clean tweets
    data_dt['clean_tweet_text'] = data_dt['words'].apply(lambda x: ' '.join(x))

    # spliting into train and test dataframes
    train_dt = data_dt[:len(train)]
    test_dt = data_dt[len(train):]
    
    # create count vectorizer with lowercase=True[default], max_features=1000 and token_pattern='[#@_$%\w\d]{2,}'
    count = CountVectorizer(max_features=1000, token_pattern='[#@_$%\w\d]{2,}')

    # transform the train dataset into bag of words
    X_train_bag_of_words = count.fit_transform(np.array(train_dt['clean_tweet_text']))

    # transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(np.array(test_dt['clean_tweet_text'])).toarray()
    
    # creating DT model with min 1% sample leaves
    dt_sentiment = DecisionTreeClassifier(
        criterion='entropy', min_samples_leaf=0.01, random_state=0)
    
    # fitting the features on the target
    dt_sentiment.fit(X_train_bag_of_words, np.array(train_dt['sentiment']))
    
    # predicting the results on test df    
    # y_pred_1 = dt_sentiment.predict(X_test_bag_of_words)
    y_pred_dt = dt_sentiment.predict_proba(X_test_bag_of_words)

    #classification reports
    # print(classification_report(y_label, y_pred_1))

    # print("--------------------------mnb")
    # preproccesing the whole dataframe for mnb
    data_mnb = preprocess(data_all, 'mnb')

    # clean_tweet_text column which contains clean tweets
    data_mnb['clean_tweet_text'] = data_mnb['words'].apply(lambda x: ' '.join(x))

    # spliting into train and test dataframes
    train_mnb = data_mnb[:len(train)]
    test_mnb = data_mnb[len(train):]

    # create count vectorizer with lowercase=True[default], max_features=1000 and token_pattern='[#@_$%\w\d]{2,}'
    count = CountVectorizer(max_features=1000, token_pattern='[#@_$%\w\d]{2,}')

    # transform the train dataset into bag of words
    X_train_bag_of_words = count.fit_transform(np.array(train_mnb['clean_tweet_text']))

    # transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(np.array(test_mnb['clean_tweet_text'])).toarray()

    # creating mnb model with all default parameters
    mnb_sentiment = MultinomialNB()

    # fitting the features on the target
    mnb_sentiment.fit(X_train_bag_of_words, np.array(train_mnb['sentiment']))
    # predicting the results on test df    
    # y_pred_2 = mnb_sentiment.predict(X_test_bag_of_words)
    y_pred_mnb = mnb_sentiment.predict_proba(X_test_bag_of_words)

    #classification reports
    # print(classification_report(y_label, y_pred_2))

    # print("--------------------------bnb")
    # preproccesing the whole dataframe for bnb
    data_bnb = preprocess(data_all, 'bnb')

    # clean_tweet_text column which contains clean tweets
    data_bnb['clean_tweet_text'] = data_bnb['words'].apply(lambda x: ' '.join(x))

    # spliting into train and test dataframes
    train_bnb = data_bnb[:len(train)]
    test_bnb = data_bnb[len(train):]

    # create count vectorizer with lowercase=True[default], max_features=1000 and token_pattern='[#@_$%\w\d]{2,}'
    count = CountVectorizer(max_features=1000, token_pattern='[#@_$%\w\d]{2,}')

    # transform the train dataset into bag of words
    X_train_bag_of_words = count.fit_transform(np.array(train_bnb['clean_tweet_text']))

    # transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(np.array(test_bnb['clean_tweet_text'])).toarray()

    # creating bnb model with all default parameters
    bnb_sentiment = BernoulliNB()

    # fitting the features on the target
    bnb_sentiment.fit(X_train_bag_of_words, np.array(train_bnb['sentiment']))

    # predicting the results on test df    
    # y_pred_3 = bnb_sentiment.predict(X_test_bag_of_words)
    y_pred_bnb = bnb_sentiment.predict_proba(X_test_bag_of_words)


    svm_sentiment = SVC()
    #classification reports
    # print(classification_report(y_label, y_pred_3))

    # print("--------------------------final")
    # integrated stacked model with soft voting
    # adding the predicted results into one result
    y_pred = y_pred_dt + y_pred_mnb + y_pred_bnb
    ans_sentiment = ['negative', 'neutral', 'positive']
    # returns the index of the max value
    y_pred = np.argmax(y_pred, axis=1)

    # get sentiment according to the index
    y_pred = pd.Series(y_pred).apply(lambda x: ans_sentiment[x])
    


    ######################################## SEABORN #######################################
    # groupedvalues = data_all.groupby(
    #     'sentiment')['instance_number'].count().reset_index()
    # print(groupedvalues.head())
    # g = sns.barplot(x='sentiment', y='instance_number', data=groupedvalues)

    # for index, row in groupedvalues.iterrows():
    #     g.text(row.name, row.instance_number, round(
    #         row.instance_number, 2), color='black', ha="center")
    # plt.title('Count of tweets by sentiment')

    # plt.tight_layout()
    # plt.savefig('q1_1.png')
    # plt.show()

    # sns.heatmap(pd.DataFrame(classification_report(
    #     y_label, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True)
    # plt.title("Integrated Stacked Model")
    # plt.tight_layout()
    # plt.savefig('mohit_model_baseline.png')
    ######################################## SEABORN #######################################

    # printing out the predicted vales / results
    for i in range(len(y_pred)):
        print(test.iloc[i]['instance_number'], y_pred[i])
    #classification reports
    # print(classification_report(y_label, y_pred))