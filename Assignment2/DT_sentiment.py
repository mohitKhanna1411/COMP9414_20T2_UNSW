# import warnings
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
# import seaborn as sns
# import matplotlib.pylab as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sys
from nltk.corpus import stopwords
# nltk.download('vader_lexicon')
# warnings.simplefilter("ignore")
# plt.style.use('ggplot')
# color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

# removes URL from the tweet using regex
# returns a string without any url


def remove_urls(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                   '', text, flags=re.MULTILINE)
    return(text)

# removes everything expect # , @, _, $ or % and numbers and characters from the tweets using regex
# returns a string without punctuation


def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\#\@\_\$\%]")
    line = rule.sub('', line)
    return line

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


def preprocess(data):
    # convert every single word into lowercase
    # data['tweet_text'] = data['tweet_text'].str.lower()

    # calling remove_urls() for each tweet
    data['tweet_text'] = data['tweet_text'].apply(lambda x: remove_urls(x))

    # calling remove_junk() for each tweet by spliting into words
    data['words'] = data['tweet_text'].apply(lambda x: remove_junk(x.split()))

    # remove stopwords from the vocabulary
    # stop_words = stopwords.words('english')
    # data['words'] = data['words'].apply(
    #     lambda x: [i for i in x if i not in stop_words])

    # stem each word from the vocabulary
    # stemmer = nltk.stem.PorterStemmer()
    # data['words'] = data['words'].apply(
    #     lambda x: [stemmer.stem(i) for i in x])

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
    # preproccesing the whole dataframe
    data_all = preprocess(data_all)
    data_all.to_csv("file1.csv", index=False)
    # clean_tweet_text column which contains clean tweets
    data_all['clean_tweet_text'] = data_all['words'].apply(
        lambda x: ' '.join(x))
    # spliting into train and test dataframes
    train = data_all[:len(train)]
    test = data_all[len(train):]

    ######################################## VADER #######################################

    # y_pred_vader = list()
    # analyser = SentimentIntensityAnalyzer()
    # for text in test['clean_tweet_text']:
    #     score = analyser.polarity_scores(text)
    #     if score['compound'] >= 0.05:
    #         # print(text+": "+"VADER positive")
    #         y_pred_vader.append('positive')
    #     elif score['compound'] <= -0.05:
    #         # print(text+": "+"VADER negative")
    #         y_pred_vader.append('negative')
    #     else:
    #         # print(text+": "+"VADER neutral")
    #         y_pred_vader.append('neutral')

    ######################################## VADER #######################################
    # create count vectorizer with lowercase=false, max_features=1000 and token_pattern='[#@_$%\w\d]{2,}'
    count = CountVectorizer(
        max_features=1000, lowercase=False, token_pattern='[#@_$%\w\d]{2,}')
    # count = CountVectorizer(
    #     max_features=1000, token_pattern='[#@_$%\w\d]{2,}')

    # transform the train dataset into bag of words
    X_train_bag_of_words = count.fit_transform(np.array(train['clean_tweet_text']))

    # transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(np.array(test['clean_tweet_text']))

    # creating DT model with min 1% sample leaves
    dt_sentiment = DecisionTreeClassifier(
        criterion='entropy', min_samples_leaf=0.01, random_state=0)

    # fitting the features on the target
    dt_sentiment.fit(X_train_bag_of_words, np.array(train['sentiment']))

    # predicting the results on test df
    y_pred = dt_sentiment.predict(X_test_bag_of_words)
    # print(dt_sentiment.predict_proba(X_test_bag_of_words))

    # ground truth
    y_label = np.array(test['sentiment'])

    # print(classification_report(y_label, np.array(y_pred_vader)))

    ######################################## SEABORN #######################################

    # sns.heatmap(pd.DataFrame(classification_report(
    #     y_label, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True)
    # plt.title('DT - After lowercase')
    # plt.tight_layout()
    # plt.savefig('dt_sentiment_lowercase.png')
    ######################################## SEABORN #######################################

    # printing out the predicted vales / results
    for i in range(len(y_pred)):
        print(test.iloc[i]['instance_number'], y_pred[i])
    # classification reports
    # print(classification_report(y_label, y_pred))
