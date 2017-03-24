import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from persistance import load_config
from utils import abort_clean, get_features_extr_name


#------------------------------------------------------------------------------
#------------------ AUTOMATED FEATURES EXTRACTORS FETCHER ---------------------
#------------------------------------------------------------------------------

def get_features_extr(features_str, verbose=1):
    '''
    Returns a list of feature extractors to match the specified features_str
    Available features extractors are :
        - cc1   : Character count (TODO)
        - wc1   : Word count - unigram (TODO)
        - wc2   : Word count - bigram
        - tfidf : TF-IDF
        - lsa   : Latent Semantic Analysis

    A feature extractor can be specified :
        - by its name --> a default clf will be instanciated
        - by a path to a config file, --> a custom clf will be instanciated
    '''
    feat_extractors = []

    #--------------------------------------------------------------------------
    # Get required features_extractor

    if(verbose):
        print("Starting loading features extractor ... ")

    if features_str == "wc2":
        feat_extractors.append(get_wc2(None))

    elif features_str == "tfidf":
        feat_extractors.append(get_wc2(None))
        feat_extractors.append(get_tfidf(None))

    elif features_str == "lsa":
        feat_extractors.append(get_wc2(None))
        feat_extractors.append(get_tfidf(None))
        feat_extractors.append(get_lsa(None))
    
    else :
        try: 
            config = load_config(features_str)
        except:
            abort_clean("Cannot load the extractors configuration",
                "Either extr name is incorrect or path is invalid : " +
                features_str)
        # Load the config from a file
        if verbose:
            print("Loading features extractor config from file ")
        feat_extractors = get_features_extr_from_file(config, verbose=verbose)

    #--------------------------------------------------------------------------
    # Return features extractors
    if(verbose):
        print("features extractor loaded: " + 
             get_features_extr_name(feat_extractors) + "\n")

    return feat_extractors


def get_features_extr_from_file(config, verbose=None):
    '''
    Returns a list of feature extractors following the given configuration
    '''
    feat_extractors = []

    # get each extractor separately 
    for extr_conf in config["extractors"]:
        if extr_conf["extractr_type"] == "wc2":
            feat_extractors.append(get_wc2(extr_conf))

        elif extr_conf["extractr_type"] == "tfidf":
            feat_extractors.append(get_tfidf(extr_conf))

        elif extr_conf["extractr_type"] == "lsa":
            feat_extractors.append(get_lsa(extr_conf))

    return feat_extractors

#------------------------------------------------------------------------------
#--------------------- FEATURES EXTRACTORS CONFIGURATORS ----------------------
#------------------------------------------------------------------------------


# Word Count (unigram and bigram)
#------------------------------------------------------------------------------
def get_wc2(config=None):
    '''
    Returns a word count (bigram) vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default wc2 setup.
    '''
    extractr_name = ""
    extractr = None

    if not (config):
        extractr_name = "wc2-default"
        extractr = CountVectorizer( #---------------------- Default Values
            input='content',
            encoding='utf-8',
            decode_error='ignore',
            strip_accents=None,
            analyzer='word',
            preprocessor=None,
            tokenizer=None,
            ngram_range=(1, 2), #--------------------- (1, 1)
            stop_words=None,
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
            max_df=1.0,
            min_df=2, #------------------------------- 1
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.int64)

    else:
        extractr_name = config["extractr_name"]
        try:
            # Adjustements due to JSON incompatibility
            config["configuration"]["ngram_range"] = tuple(
                config["configuration"]["ngram_range"] )
            config["configuration"]["dtype"] = np.int64

            extractr = CountVectorizer(**(config["configuration"]))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " + 
                config["configuration"])

    res = (extractr_name, extractr)
    return res


# Term Frequency - Inverse Document Frequency
#------------------------------------------------------------------------------
def get_tfidf(config=None):
    '''
    Returns a tfidf vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default tfidf setup.
    '''
    extractr_name = ""
    extractr = None

    if not (config):
        extractr_name = "tfidf-default"
        extractr = TfidfTransformer(
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False)

    else:
        extractr_name = config["extractr_name"]
        try:
            extractr = TfidfTransformer(**(config["configuration"]))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " + 
                config["configuration"])

    res = (extractr_name, extractr)
    return res


# Latent Semantic Analysis
#------------------------------------------------------------------------------
def get_lsa(config=None):
    '''
    Returns a latent semantic analysis vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default lsa setup.
    '''
    extractr_name = ""
    extractr = None

    if not (config):
        extractr_name = "lsa-default"
        extractr = TruncatedSVD( #------------------------- Default Values
            n_components=1000, #---------------------- 2
            algorithm="randomized",
            n_iter=10,
            random_state=42,
            tol=0.
        )

    else:
        extractr_name = config["extractr_name"]
        try:
            extractr = TruncatedSVD(**(config["configuration"]))
        except:
            abort_clean("Features Extractor configuration failed",
                "Configuring " + config["extractr_type"] + " with : " + 
                config["configuration"])

    res = (extractr_name, extractr)
    return res
