# Lab 2
# Group Members: Lucy Kien, Ken Howry, Brianna Brost
# April 3rd 2024
# Prof. GauthierDickey

import re
from nltk import PorterStemmer


# Part 2
# 1
def find_the(text):
    # find all points of the word 'the'
    result = re.findall(r'\bthe\b', text)
    return result

# 2
def find_hello(text):
    # find hello while ignoring the case sentivity
    result = re.search(r'hello', text, re.IGNORECASE)
    if result:
        # return the result
        return result.group()
    else:
        # else none
        return None

# 3
def find_digit(string):
    digits = re.findall(r'\d{1,}', string)
    return digits


# 4
def find_vowel_words(string):
    # find all words that begin with a vowel 
    vowel_words = re.findall(r'\b[aeiou]\w*', string, re.IGNORECASE)
    return vowel_words


# 5
def find_n_k(word, number):
    # split the words
    l = word.split()
    # make a result list
    result = []
    # for all words in the string 
    for w in l:
        # if there is an n and it is greater than or equal to the number specified 
        if w.count('n') >= number:
            # append to the list
            result.append(w)
    return result

# 6
#returns a list of the words that start and end with the same letter
def match_start_end(text: str):
    words = re.findall(r'\b(\w+(\w)\w*)\b', text.lower())
    result = [word[0] for word in words if word[1] == word[0][0]]
    print('Words that start and end with the same letter:', result)

# 7
def first_word_in_sentence(text):
    # reg expression to find the first word of each sentence
    pattern = r'(?:[.!?]\s*)([A-Z][a-z]*)'
    first_words = re.findall(pattern, text)
    # base case to catch the first word in the string. This will be the start of the first sentence
    if text[0].isupper(): 
        first_words.insert(0, text.split()[0])
    return first_words


# 8
def valid_date(date):
    try:
        # seperate the date out
        year, month, day = map(int, date.split('-'))
        if len(str(year)) != 4:
            # more than 4 ints in year false
            return False
        if month < 1 | month > 12:
            # not within scope of months
            return False
        if day < 1:
            # negatives
            return False
        if month in {1, 3, 5, 7, 8, 10, 12}:
            # months with 31 days
            return day <= 31
        elif month in {4, 6, 9, 11}:
            # months with 30 days
            return day <= 30
        elif month == 2:
            # leap year function below
            if is_leap_year(year):
                return day <= 29
            else:
                return day <= 28
        else:
            return False
    except ValueError:
        return False
    
# leap year function used an LLM to figure this one out
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# 9
def extract_domain(url):
    # take away the first part
    if url.startswith('https://'):
        url = url[len('https://'):]  
    # split it by the /
    parts = url.split('/') 
    # extract the domain that directly follows the //
    domain = parts[0].split('//')[-1]
    # return the domain
    return domain


# 10
def remove_RT(string):
    # subs out RT with an empty string and returns the new string
    # only if it starts with RT
    result = re.sub(r'^RT\s', '', string, flags=re.IGNORECASE)
    return result


# 11
def extract_IP_addresses(string):
    # reg expression result where d1-d4 are between 1-3 digits
    result = r'\b(?:\d{1,3}\.){3}\d{1,3}\b' 
    # use find all to find the addresses in the string given 
    ip_addresses = re.findall(result, string)
    # result list for all ips found
    valid_ips = []
    for ip in ip_addresses:
        # split the ip into parts by '.'
        parts = ip.split('.')
        # make sure there are four seperate objects
        if len(parts) != 4:
            continue  
        # check for the range
        valid = True
        for part in parts:
            if not (0 <= int(part) <= 255):
                valid = False
                break
        # if in the range append to the list
        if valid:
            valid_ips.append(ip)
    
    # return the list
    return valid_ips

# 12
def replace_color(sentence):
    # sub the word color with colour and output new sentence
    sentence = re.sub(r'\bcolor\b', 'colour', sentence)
    return sentence

# Part 3

# now create the stemmer
stemmer = PorterStemmer()

def stem_words(string):
    stemmed = []
    for word in string:
        stemmed.append(stemmer.stem(word))
    return stemmed

def main():
    print("Question 1")
    print(find_the("the point at which the object is"))
    print("")
    print("Question 2")
    print(find_hello("theHELLohi"))
    print(find_hello("hello HELLO"))
    print("")
    print("Question 3")
    print(find_digit("Hey 230 how are you doing today, 7."))
    print("")
    print("Question 4")
    print(find_vowel_words("hey ello dude how are you. I love it."))
    print("")
    print("Question 5")
    # should output ['manners', 'yoink']
    print(find_n_k("manners yoink", 1))  
    # shoult output since looking for two consecutive n ['manners']
    print(find_n_k("manners yoink", 2))  
    print("")
    print("Question 6")
    match_start_end('The eve of tent seeds')
    print("")
    print("Question 7")
    print(first_word_in_sentence("How are you doing. What is up. How are you."))
    print("")
    print("Question 8")
    # leap year true
    print(valid_date("2024-02-29")) 
    #  false not a leap year
    print(valid_date("2021-02-29")) 
    # true normal year
    print(valid_date("2024-10-03"))
    print("")
    print("Question 9")
    # example domain where it will give canvas.du.edu
    print(extract_domain('https://canvas.du.edu/courses/')) 
    # without hhtps still finds the domain
    print(extract_domain('canvas.du.edu/courses/'))
    print("")
    print("Question 10")
    # no print
    print(remove_RT("RT don't even think about it"))
    # yes print
    print(remove_RT("I would always print RT")) 
    print("")
    print("Question 11")
    print(extract_IP_addresses("This is an IP address: 0.0.0.0"))
    print(extract_IP_addresses("This is an IP address: 255.255.255.255"))
    print(extract_IP_addresses("This is an IP address: 3.22.214.0"))
    print(extract_IP_addresses("This is not an IP address: 3.255.-1.2"))
    print(extract_IP_addresses("This is multiple IPs: 0.0.0.0 233.25.1.7"))
    print("")
    print("")
    print("Question 12")
    print(replace_color("Hey what is your favorite color?"))
    print("Part 3")
    print(f'The stem of happy is {stemmer.stem("happy")}')
    words = ["happy", "finally", "continue", "friendly", "biology", "value", "likely"]
    print(stem_words(words))

if __name__ == main():
    main()