'''
File: tweet_processor.py
Author: Lucy Kien
Date: April 10th, 2024

Python module to handle the processing of tweets. Generally, we use either
process_tweet or process_tweets.
'''

import re
import json
import string
import random

# import tweet tokenizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer 


# read a json file and store the tweets from it as a list of strings
def load_tweets(filename: str) -> list[str]:
    '''
    Read a json file and returns a list of
    strings representing the tweets.

    Parameters:
    filename -- the name of the json file to read, this assumes that
    each line of the file is a complete json object that's a tweet
    '''
    tweets = []
    # read the file line by line and use json.loads to read it in
    # note that the json object will have a .text field which is the 
    # actual content of the tweet--and what we want to return 
    # read the file line by line
    # open the file
    with open(filename, 'r') as file:
        # for each line in the file use json to load it
        for line in file:
            tweet_data = json.loads(line)
            # append it to the tweets list 
            tweets.append(tweet_data['text'])
    # now return the tweets as a list
    return tweets

# cleanup the tweets
def cleanup_tweet(tweet : str) -> str:
    '''
    Walk through a tweet and remove:
      1) the text RT at the start of a tweet is removed (this is an 
         old way that 'retweet' was indicated); 
      2) URLs are removed, which begin with http:// or https://, 
         and have some domain name; and 
      3) the hash of a hash-tag. 
      
    Parameters:
      tweets -- the list of tweets, which is a list of strings

    Return:
      This function doesn't return anything but modifies the tweets in place.
    '''
    # remove RT at the start of the tweet
    cleaned_tweet = re.sub(r'^RT', '', tweet)
    # remove URLs starting with http:// or https://
    cleaned_tweet = re.sub(r'https?://\S+', '', cleaned_tweet)
    # remove hashtags
    cleaned_tweet = re.sub(r'#', '', cleaned_tweet)

    return cleaned_tweet

# tokenize the tweets
def tokenize_tweet(tweet : str, tokenizer : TweetTokenizer) -> list[str]:
    '''
    Converts a tweets to a list of a list of tokens, each a string

    Paramters:
    tweet -- a string representing the tweet

    Returns:
    A list of tokens, where each element is a string
    '''
    # using the tokenizer to tokenize the tweet
    tokens = tokenizer.tokenize(tweet)
    # return the tweet with the tokenized version
    return tokens

# here we remove the stopwords and punctuation
def remove_stopwords_and_punctuation(tweet_toks : list[str], 
                                     stopwords : list[str], 
                                     punctuation : list[str]) -> list[str]:
    '''
    Take a list of strings and return a list with the stop words and
    punctuation removed from it.

    Parameters:
      tweet -- a list of tokens, each a string
      stopwords -- a list of stopwords
      punctuation -- a list of punctuation
    '''
    newtweet_toks = []
    # walk through the tokens in the tweet and remove any that are
    # stopwords or punctuation from thie list of tokens
    for token in tweet_toks:
        # Check if the token is not a stopword and not punctuation
        if token not in stopwords and token not in punctuation:
            # If not, add it to the new list of tokens
            newtweet_toks.append(token)
    # return the new list of tokens
    return newtweet_toks

# takes a list of tweets and returns the tweets with the stemmed words,
# note that the argument is a list of a list of strings, since each element
# of a tweet contains a list of tokens in that tweet
def stem_tweet(tweet_toks : list[str], stemmer : PorterStemmer) -> list[str]:
    '''
    Take a list of tweet tokens and stem each token, which will replace
    a word with its stem (and possibly just keep it if we can't stem it)

    Paramters:
      tweet_toks -- a list of tweet tokens
      stemmer -- a PorterStemmer object for stemming
    '''
    stemmed_toks = []
    # walk through the list of tokens passed in and stem them using
    # the stemmer parameter (note, there is a stem function on it)
    for token in tweet_toks:
        stemmed_toks.append(stemmer.stem(token))
    return stemmed_toks

# read and return a list of stopwords
def parse_stopwords(filename : str) -> list[str]:
    # read our stopwords from a file and return them as a list of strings
    stopwords = []
    with open(filename, 'r') as file:
        # Read stopwords from the file and split by newline
        stopwords = file.read().splitlines()
    return stopwords

# parse and load the tweets
def process_tweets(pos_name : str, neg_name : str, stopwords_name : str) -> tuple[(list[str], list[str], list[str])] :
    '''
    process_tweets takes three arguments that are file names of
    positive tweets, negative tweets, and stopwords. It then cleans
    up the tweets by removing stop words, lowercasing, tokenizing
    and stemming the words in the tweets.

    Parameters:
    pos_name -- the file name of the positive tweet set
    neg_name -- the file name of the negative tweet set
    stopwords_name -- the file name of the stopwords list

    Returns:
    Three values, a list of strings of the positive tweets, a list of strings
    of the negative tweets, and a list of strings of the stopwords 
    '''
    # punctuation words
    punctuation = [',', "'", '?', ".", "!", ";", ":", "&", "...", "(", ")", "/", ":(", ":)", ":-(", "-", ">:(", "xD", ":p", ".."]
    
    # read the samples as json files and get the list of 
    # positive and negative tweets from each file
    pos_tweets = load_tweets(pos_name)
    neg_tweets = load_tweets(neg_name)

    # now create a tokenizer and porter stemmer so we can reuse 
    # them each function call when calling process tweets
    tokenizer = TweetTokenizer(preserve_case = False, 
                               strip_handles = True,
                               reduce_len=True)
    stemmer = PorterStemmer()

    # now process each tweet and create a new list of processed positive and negative tweets

    # Process positive tweets
    processed_pos_tweets = []
    for tweet in pos_tweets:
        processed_tweet = process_tweet(tweet, stopwords_name, punctuation, tokenizer, stemmer)
        processed_pos_tweets.append(processed_tweet)

    # Process negative tweets
    processed_neg_tweets = []
    for tweet in neg_tweets:
        processed_tweet = process_tweet(tweet, stopwords_name, punctuation, tokenizer, stemmer)
        processed_neg_tweets.append(processed_tweet)
    
    # now return the processed tweets and any used stopwords
    return processed_pos_tweets, processed_neg_tweets, stopwords_name


def process_tweet(tweet : str, 
                  stopwords: list[str],
                  punctuation = string.punctuation,
                  tokenizer = TweetTokenizer(preserve_case = False, 
                                             strip_handles = True,
                                             reduce_len=True),
                  stemmer = PorterStemmer()) -> list[str]:
    '''
    Processes an individual tweet, returning its stemmed version.

    Parameters: 
      tweet -- a string representing the tweet
      tokenizer -- A tokenizer object with type TweetTokenizer. We default
        to preserving case (so that all words are converted to lowercase),
        stripping handles, and reducing the length of the tweet. Reducing
        the lenght means that any characters repeated more than 3 times will
        be reduced to 3 characters. So Hiiiiii would be Hiii. 
      stemmer -- a stemmer object that takes a string and returns its stem,
        if there is one, or the same string back otherwise. The default is
        the PorterStemmer from nltk. The object requires a stem method that
        takes a string and returns the stem of the string.

    Return: a list of tokens that have been processed
    '''
    # first cleanup the tweet
    cleaned_tweet = cleanup_tweet(tweet)
    
    # then tokenize
    token = tokenize_tweet(cleaned_tweet, tokenizer)

    # then remove stopwords and punctuation
    remove_stop_punc = remove_stopwords_and_punctuation(token, stopwords, punctuation)
    
    # finally stem the tweet
    stemmed_tokens = stem_tweet(remove_stop_punc, stemmer)

    # and return the result
    return stemmed_tokens

def test_tweet_processing():
    tweets = ["I am happy", "I am sad", "RT: Hi there", "Hi #happy #sad!", "You should go to DU: https://www.du.edu!"]
    stopwords = parse_stopwords('TweetProcessor/english_stopwords.txt')
    for tweet in tweets:
        print(f'processing {tweet}: {process_tweet(tweet, stopwords)}')

    # try processing all of the positive and negative tweets now!
    print(f'processing all of the tweets now...')
    pos_tweets, neg_tweets, stopwords = process_tweets('TweetProcessor/positive_tweets.json', 'TweetProcessor/negative_tweets.json', 'TweetProcessor/english_stopwords.txt')
    print(f'done!')

    print(f'random positive tweet: {pos_tweets[random.randint(0, len(pos_tweets) - 1)]}')
    print(f'random negative tweet: {neg_tweets[random.randint(0, len(neg_tweets) - 1)]}')
    print(f'random stopword: {stopwords[random.randint(0, len(stopwords) - 1)]}')
    


def main():
    test_tweet_processing()

if __name__ == '__main__':
    main()
