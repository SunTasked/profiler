import html
from os import listdir, walk
from os.path import isfile, join
import re
from time import time
import xml.etree.ElementTree as ET

from utils import get_printable_tweet


#------------------------------------------------------------------------------
#------------------------------ PARSING MODULE --------------------------------
#------------------------------------------------------------------------------

def process_text(tweet):
    '''
    Processes the text given as parameter and removes the following elements:
        - URLS
        - @users
    '''

    # html unescape special chars.
    if tweet:
        tweet = html.unescape(tweet)
    else:
        return u""
        
    # filters
    tweet = re.sub(u'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(u'@[^\ ]*', '', tweet, flags=re.MULTILINE)
    return str(tweet)


def parse_file(file_to_parse, file_to_save=None, verbose=False) :
    '''
    Takes a input file name.
    Parses all the tweets it contains.
    If specified, the results of the parsing will be stored into file_to_save
    Returns an object containing :
        - "lang" : a string representing the tweets language
        - "tweets" : a table containing the tweets (utf8 encoding)
    '''

    # Parsing PAN 2017 xml formatted documents:
    # <author lang="XX">
    #   <documents>
    #       <document>TWEET</document>
    #   </documents>
    # </author>
    try:
        tree = ET.parse(file_to_parse)    
        root = tree.getroot()
    except:
        if verbose:
            print("file " + file_to_parse + " : PARSING ERROR")
        return None, []

    author_attr = root.attrib
    documents = root.findall("./documents/document")

    tweets = []
    for idx, doc in enumerate(documents) :
        
        processed_tweet = process_text(doc.text)
        doc.text = processed_tweet
        tweets.append(processed_tweet)
        
        if (verbose):
            print (file_to_parse, idx, get_printable_tweet(processed_tweet))
    
    if (file_to_save):
        tree.write(file_to_save, encoding="utf8")

    res = { "lang" : author_attr["lang"],
            "tweets" : tweets }
    return res


def filter_tweets (author, verbose=False):
    '''
    Removes some tweets given their caracteristics :
        - short tweets (less than 5 chars)
    Returns the filtered tweet list
    '''

    tweets = author["tweets"]
    n_init_tweets = len(tweets)
    if verbose: print ("      - Initial tweets = " + str(n_init_tweets))

    # filter empty tweets
    tweets = [t for t in tweets if len(t)>5]
    n_empty_tweets = n_init_tweets - len(tweets)
    if verbose: print ("      - Empty tweets = " + str(n_empty_tweets))

    # HERE : insert other filter

    if verbose: print ("      - Saved tweets = " + str(len(tweets)))

    return tweets


def parse_tweets_from_dir(input_dir, output_dir=None, verbosity_level=1) :
    '''
    Parses all the xml files directly in the input_dir (no recursion).
    Retrieves the attributes of the author stored in the truth file.
    If specified, the parsed files will be written into the output_dir
    Verbosity level specifies the amount of content displayed:
        0- nothing
        1- Main steps
        2- Files parsed and stats about filtering / tweets available per class
        3- All the parsed content.
    Returns a list containing all the author objects contained within the 
    input_dir
    '''

    # vars
    Authors = []
    t0 = time()
    n_files = 0
    n_files_parsed = 0
    n_files_filtered = 0
    n_files_infos_retrieved = 0
    ret = '\n'

    # preprocessing on direcory paths
    if input_dir[-1] != "/": input_dir = input_dir + "/"
    if output_dir and output_dir[-1] != "/": output_dir = output_dir + "/"

    # ---------------------------- FILES LISTING
    
    if verbosity_level:
        t0 = time()
        print ("Starting files Listing ...")
    try:
        xml_files = [f for f in listdir(input_dir) if (
            isfile(join(input_dir, f)) and f[-4:] == ".xml" )]
    except:
        print("Files listing --- failure")
        print("Maybe the directory specified is incorrect ?")
        return

    if verbosity_level:
        print("Files found : " + str(len(xml_files)))
        print("Files listing --- success in %.3f seconds\n"  %(time()-t0))


    # ---------------------------- FILES PROCESSING
    if verbosity_level:
        t0 = time()
        print ("Starting files processing ...")
    
    n_files = len(xml_files)

    for f in xml_files :
        author = None
        tweets = []
        save_file = output_dir + f if output_dir else None
        try:
            author = parse_file(input_dir + f, save_file, verbosity_level > 2)
        except:
            if verbosity_level > 1:
                print("   Parsing file : " + f + " --- failure")
            continue

        if verbosity_level > 1:
            print("   Parsing file : " + f + " --- success")

        n_files_parsed += 1
        author["id"]= f[:-4]
        Authors.append(author)

    if verbosity_level :
        print("Parsed files : " + str(n_files_parsed) + " out of " + 
            str(n_files))
        print("Files Parsing --- success in %.3f seconds\n"  %(time()-t0))


    # ---------------------------- AUTHOR ATTRIBUTES RETRIEVING

    if verbosity_level:
        t0 = time()
        print ("Starting Author Attributes Retrieval ...")
    try:
        truth_file = open(input_dir + "truth.txt")
    except:
        print("Author Attributes Retrieval --- failure")
        print("Couldn't open " + input_dir + "truth.txt")
        return
    
    truth_lines = [x.strip().split(':::') for x in truth_file.readlines()]
    attrs = dict()
    for l in truth_lines :
        attrs[l[0]] = l[1:]

    for idx, author in enumerate(Authors):
        author["gender"] =  attrs[author["id"]][0]
        author["variety"] = attrs[author["id"]][1]

        if author["gender"] and author["variety"]:
            n_files_infos_retrieved += 1

        if verbosity_level > 1:
            print("   author " +Authors[idx]["id"] +
                  " information retrieved : Gender=" + Authors[idx]["gender"] +
                  " Language=" + Authors[idx]["lang"] + 
                  " Variety=" + Authors[idx]["variety"])

    if verbosity_level > 1:
        print("Retreived Information : " + str(n_files_infos_retrieved) + 
            " out of " + str(n_files))
        print("Author Attributes Retrieval --- success in %.3f seconds\n"  
            % (time() - t0))


    # ---------------------------- TWEET FILTERING:
    if verbosity_level :
        t0 = time()
        print("Starting Tweets Filtering ...")

    for author in Authors :

        if verbosity_level > 1:
            print ("   author " + author["id"] + " filtering")

        try:
            author["tweets"] = filter_tweets(author, verbosity_level>1)
        except:
            continue
        
        n_files_filtered += 1

    if verbosity_level :
        print("Filtered files : " + str(n_files_filtered) + " out of " + 
            str(n_files))
        print("Tweets Filtering --- success in %.3f seconds\n"  
            % (time() - t0))


    # ---------------------------- RETURNING PROCESSED DATA

    if verbosity_level :
        print("Tweets available for learning : " + 
            str(sum([len(a["tweets"]) for a in Authors])) + "\n")

    return Authors


def parse_tweets_from_main_dir (input_dir, output_dir=None, verbosity_level=1) :
    '''
    Parses all the xml files contained within directories included in the 
    specified input_dir (1 level recursion).
    Verbosity level specifies the amount of content displayed:
        0- nothing
        1- Main steps
        2- Files parsed and stats about filtering / tweets available per class
        3- All the parsed content.
    Returns a list containing all the author objects contained within the 
    input_dir 
    '''

    # vars
    Authors = []
    t0 = time()
    n_files = 0
    n_files_parsed = 0

    # preprocessing on direcory paths
    if input_dir[-1] != "/": input_dir = input_dir + "/"
    if output_dir and output_dir[-1] != "/": output_dir = output_dir + "/"

    subdirs = next(walk(input_dir))[1]
    if verbosity_level :
        print("--------------------------------------")
        print("Starting Parsing of main directory ...")
        print("--------------------------------------\n")

    for sub in subdirs:
        if verbosity_level :
            print("--------------------------------------")
            print("Parsing subdirectory : " + sub + "\n")
        if output_dir :
            Authors = Authors + parse_tweets_from_dir(input_dir+sub, 
                output_dir+sub, verbosity_level)
        else :
            Authors = Authors + parse_tweets_from_dir(input_dir+sub, 
                None, verbosity_level)
        if verbosity_level :
            print("--------------------------------------\n")

    if verbosity_level :
        print("--------------------------------------")
        print("Total tweets available for learning : " + 
            str(sum([len(a["tweets"]) for a in Authors])))
        print("Parsing of main directory --- success in %.3f seconds"  
            % (time() - t0))
        print("--------------------------------------")

    return Authors