import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils import abort_clean


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
        - lsa   : Latent Semantic Analysis (TODO)

    A feature extractor can be specified :
        - by its name --> a default clf will be instanciated
        - by a path to a config file, --> a custom clf will be instanciated (TODO)
    '''

    #--------------------------------------------------------------------------
    # Check basic requirements


    if(verbose):
        print("Starting loading features extractor ... ")

    
    #--------------------------------------------------------------------------
    # Get required features_extractor
    feat_extractors = []

    if features_str == "wc2":
        feat_extractors = get_wc2()

    elif features_str == "tfidf":
        feat_extractors = get_tfidf()

    elif features_str == "lsa":
        #abort_clean("LSA is unstable for now")
        feat_extractors = get_lsa()
    
    else :
        abort_clean("Unknown features extractor.")
    
    #--------------------------------------------------------------------------
    # Return features extractors
    if(verbose):
        print("features extractor loaded: '" + 
            "' + '".join([x[0] for x in feat_extractors]) + "'\n")
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

    if not (config):
        wc2 = CountVectorizer( #---------------------- Default Values
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
        
        wc2_name = "wc2-default"
        res = [(wc2_name, wc2)]
        return res


# Term Frequency - Inverse Document Frequency
#------------------------------------------------------------------------------
def get_tfidf(config=None):
    '''
    Returns a tfidf vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default tfidf setup.
    '''

    if not (config):
        wc2 = get_wc2()
        tfidf_transform = TfidfTransformer(
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False)

        tfidf_transform_name = "tfidf-default"
        res = wc2 + [(tfidf_transform_name, tfidf_transform)]
        return res


# Latent Semantic Analysis
#------------------------------------------------------------------------------
def get_lsa(config=None):
    '''
    Returns a latent semantic analysis vectorizer.
    If specified, follows the config to setup the vectorizer
    Else follows default lsa setup.
    '''

    if not (config):
        tfidf = get_tfidf()
        lsa = TruncatedSVD( #------------------------- Default Values
            n_components=1000, #---------------------- 2
            algorithm="randomized",
            n_iter=10,
            random_state=42,
            tol=0.
        )

        lsa_name = "lsa-default"
        res = tfidf + [(lsa_name, lsa)]
        return res
