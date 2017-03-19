from twython import Twython
import html
import xml.etree.ElementTree as ET
import re
import time
from os import listdir
from os.path import isfile, join


CONSUMER_KEY = "key"
CONSUMER_SECRET = "secret"
OAUTH_TOKEN = "token"
OAUTH_TOKEN_SECRET = "secret_token"
twitter = Twython(
    CONSUMER_KEY, CONSUMER_SECRET,
    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)


# number of tweets downed since last pause
ntweets_sincelastpause = 0
nrequests = 0
# total number of tweets downed
ntweets_downloaded = 0
# time of last pause
time_pause = time.time()



def download_tweet(tweet_id):
    global ntweets_downloaded
    global ntweets_sincelastpause
    global nrequests

    if (nrequests == 898):
        print("pausing for 15min 30sec")
        print("   - tweets dwnl since last pause :      " + 
            str(ntweets_sincelastpause))
        print("   - total tweets dwnl since beggining : " + 
            str(ntweets_downloaded))
        time.sleep(930)
        ntweets_sincelastpause = 0
        nrequests = 0

    tweet = ""
    
    try :
        tweet = twitter.show_status(id=tweet_id)["text"]
    except :
        tweet = ""
    
    if (tweet):
        ntweets_sincelastpause += 1
        ntweets_downloaded += 1

    nrequests +=1
    return html.unescape(tweet)


def get_printable_tweet(tweet_text):
    '''
    As many utf8 caracters are not convertible to ascii/charmap, this function
    removes unprintable caracters for the console.
    '''
    return re.sub(u'[^\x00-\x7f]',u'', tweet_text)


def download_file(file_to_parse, file_to_save, verbose=False) :
    '''
    This function takes a input file and parses all the tweets it contains.
    If specified, the function will additionnaly store the download tweet 
    into the file_to_save
    Returns the author attributes and all the tweets into a table of strings
    (respecting utf8 encoding)
    '''

    try:
        tree = ET.parse(file_to_parse)    
        root = tree.getroot()
    except:
        if verbose:
            print("file " + file_to_parse + " : PARSING ERROR")
        return None, []

    author = root.attrib

    tweets = []

    # document must respect PAN formatting
    documents = root.findall("./documents/document")

    i =0
    for doc in documents :
        # get tweet id 
        tweet_id = doc.attrib["id"]
        #download tweet using it's id
        tweet_text = download_tweet(tweet_id)

        doc.text = tweet_text
        tweet = {"id" : tweet_id,
                "text" : doc.text}

        tweets.append(tweet)

        i+=1
        if (verbose):
            print (file_to_parse, i, get_printable_tweet(tweet_text))
    
    tree.write(file_to_save, encoding="utf8")

    return author, tweets


def filter_tweets (author, verbose=False):
    '''
    Filters the tweets on some given parameters
    This function is bound to evolve given that some filtering option might 
    not be adapted to the different learning algorithms
    Returns the filtered tweet list
    '''

    tweets = author["tweets"]
    n_init_tweets = len(tweets)
    # remove empty tweets
    tweets = [t for t in tweets if len(t["text"])]
    n_empty_tweets = n_init_tweets - len(tweets)
    if verbose:
        print ("      - Empty tweets = " + str(n_empty_tweets))

    # remove duplicates (optimized)
    seen = set()
    seen_add = seen.add
    tweets = [t for t in tweets if not (t["text"] in seen or 
        seen_add(t["text"]))]
    n_duplicates = n_init_tweets - n_empty_tweets - len(tweets)
    if verbose:
        print ("      - Duplicate tweets = " + str(n_duplicates))

    return tweets


def download_tweets (input_dir, output_dir=None, verbosity_level=1) :
    '''
    parse all the files in the input_dir.
    If specified, the parsed files will be stored into the output_dir
    Verbosity level specifies the amount of content displayed:
        0- nothing
        1- Main steps
        2- Files parsed and stats about filtering / tweets available per class
        3- All the parsed content.
    '''

    # vars
    Authors = []
    t0 = time.time()
    n_files = 0
    n_files_downloaded = 0
    ret = '\n'

    # preprocessing on direcory paths
    if input_dir[-1] != "/":
        input_dir = input_dir + "/"
    if output_dir and output_dir[-1] != "/":
        output_dir = output_dir + "/"

    
    
    # ---------------------------- FILES LISTING
    
    if verbosity_level:
        t0 = time.time()
        print ("Starting files Listing ...")
    try:
        xml_files = [f for f in listdir(input_dir) if 
            (isfile(join(input_dir, f)) and f[-4:] == ".xml")]
    except:
        print("Files listing --- failure")
        print("Maybe the directory specified is incorrect ?")
        return

    if verbosity_level:
        print("Files found : " + str(len(xml_files)))
        print("Files listing --- success in %.3f seconds\n"  
            % (time.time() - t0))


    # ---------------------------- FILES PROCESSING
    if verbosity_level:
        t0 = time.time()
        print ("Starting files processing ...")
    
    n_files = len(xml_files)

    for f in xml_files :
        author = None
        tweets = []
        save_file = output_dir + f if output_dir else None
        try:
            author, tweets = download_file(input_dir + f, 
                save_file, verbosity_level > 2)
        except:
            if verbosity_level > 1:
                print("   Parsing file : " + f + " --- failure")
            continue

        if verbosity_level > 1:
            print("   Parsing file : " + f + " --- success")

        n_files_downloaded += 1
        author["tweets"] = tweets
        author["id"]= f[:-4]
        Authors.append(author)

    if verbosity_level :
        print("Parsed files : " + str(n_files_downloaded) + 
            " out of " + str(n_files))
        print("Files Parsing --- success in %.3f seconds\n"  
            % (time.time() - t0))


download_tweets("./tweets/origin/", "./tweets/processed/", verbosity_level=3)
print("process ended.")
print("   - total tweets dwnl since beggining : " + str(ntweets_downloaded))