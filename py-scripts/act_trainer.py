from time import time

from numpy import array
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

from classifiers import get_classifier
from features import get_features_extr
from persistance import save_model, save_scores
from dataset_parser import parse_tweets_from_main_dir, parse_tweets_from_dir
from pipeline import get_pipeline
from utils import build_corpus, abort_clean, print_scores
from utils import get_classifier_name, get_features_extr_name


#------------------------------------------------------------------------------
#------------------------------ TRAINING MODULE -------------------------------
#------------------------------------------------------------------------------

def train(options) :
    '''
    Trains a specified classifier on a specified dataset using specified feats
    Will proceed as follows :
        - loads the dataset
        - builds the corpus
        - loads the classifier
        - loads the features extractor
        - builds the execution pipeline
        - trains the classifier on the corpus
        - cross-validates the resulting model [optional]
        - saves the resulting model [optional]
    '''

    #--------------------------------------------------------------------------
    # Check basic requirements
    if not (options["labels"]):
        abort_clean("Labels not specified", "expected 'l', 'g' or 'v'")
    
    if not (options["features"]):
        abort_clean("Features not specified")

    if not (options["classifier"]):
        abort_clean("Classifier not specified")


    #--------------------------------------------------------------------------
    # Load the tweets
    if 'l' in options["labels"] or "language" in options["labels"]: 
        # load all tweets for language classification
        Authors = parse_tweets_from_main_dir(
            input_dir=options["input-dir"], 
            output_dir=options["processed-tweets-dir"],
            verbosity_level=options["verbosity"])
    else : 
        # load tweets in one language for variety or gender classification
        Authors = parse_tweets_from_dir(
            input_dir=options["input-dir"], 
            output_dir=options["processed-tweets-dir"],
            label=True,
            verbosity_level=options["verbosity"])

    if not (Authors):
        abort_clean("Tweets loading failed")


    #--------------------------------------------------------------------------
    # Build the corpus and label the tweets
    corpus, labels = build_corpus(
        authors=Authors, 
        labels=options["labels"],
        shuffle=False, 
        verbosity_level=options["verbosity"])

    if corpus.empty or not(labels):
        abort_clean("Corpus building failed")
    

    #--------------------------------------------------------------------------
    # Load the classifier
    
    t0 = time()
    classifier = get_classifier(
        classifier_str=options["classifier"][0],
        config=None,
        verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Load the features extractors

    features_extr = get_features_extr(        
        features_str_list=options["features"][0],
        verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Build the execution pipeline

    pipeline = get_pipeline(
        features_extr=features_extr, 
        classifier=classifier, 
        verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Train the execution pipeline

    # train and cross validate results
    if (options["cross-validation"]):
        if(options["verbosity"]):
            print("Model Training with cross validation")

        pipeline, scores = train_model_cross_validation(
                            corpus=corpus, 
                            labels=labels, 
                            pipeline=pipeline, 
                            verbose=options["verbosity"])
        
        if options["verbosity"]:
            print_scores(scores)
        if options["output-dir"]:
            filename = str(get_features_extr_name(features_extr) + 
                "+" + get_classifier_name(classifier))
            save_scores(
                scores=scores,
                output_dir=options["output-dir"],
                filename=filename,
                verbose=options["verbosity"])

    # train without validation --> output-dir required
    else:
        if options["verbosity"]:
            print("Model Training without cross validation")
        if not(options["output-dir"]):
            abort_clean(
                "No output directory specified.", 
                "Training without persisting is not allowed")
        pipeline = train_model(
                    corpus=corpus,
                    pipeline=pipeline, 
                    verbose=options["verbosity"])


    #--------------------------------------------------------------------------
    # Save the resulting model
    filename = str(get_features_extr_name(features_extr) + 
            "+" + get_classifier_name(classifier))
    save_model(
        pipeline=pipeline, 
        output_dir=options["output-dir"],
        filename=filename,
        verbose=options["verbosity"]) 


    #--------------------------------------------------------------------------
    # End Execution
    if options["verbosity"]:
        print("Training task complete in " + str(round(time()-t0)) + " s")



#------------------------------------------------------------------------------
#------------------------- MODEL TRAINING FUNCTIONS ---------------------------
#------------------------------------------------------------------------------

def train_model(corpus, pipeline, verbose):
    '''
    Takes a pipeline and train it on the specified corpus.
    Returns the trained pipeline once finished.
    '''

    if verbose :
        t0 = time()
        print("Starting model training ... (this may take some time)")

    # retrieve tweets and labels
    train_text = corpus['tweets'].values
    train_y = corpus['class'].values

    # train the pipeline
    pipeline.fit(train_text, train_y)
    
    if verbose :
        print("Model training complete in %.3f seconds\n"  % (time() - t0))

    return pipeline


def train_model_cross_validation(corpus, labels, pipeline, verbose=1):
    '''
    Takes a pipeline and train it on the specified corpus.
    Processes a cross-validation algorithm (K-fold) in order to evaluate the
    quality of the model.
    Returns the best trained pipeline (in terms of macro f-score).
    '''

    if verbose :
        t0 = time()
        print("Starting model Cross Validation ... (this may take some time)")

    confusion = array(
        [[0 for x in range(len(labels))] for y in range(len(labels))])
    scores = []
    best_f_score = 0
    best_pipeline = None
    scores_micro=[]
    scores_macro=[]

    # start Kfold cross validation.
    n_run = 1
    k_fold = KFold(n_splits=10, shuffle=True)
    for train_indices, test_indices in k_fold.split(corpus):
        # train model
        train_corpus = corpus.iloc[train_indices]
        pipeline = train_model(
            corpus=train_corpus, 
            pipeline=pipeline,
             verbose=0)

        # test model 
        test_text = corpus.iloc[test_indices]['tweets'].values
        test_y = corpus.iloc[test_indices]['class'].values
        predictions = pipeline.predict(test_text)
        
        # compute metrics
        confusion += confusion_matrix(test_y, predictions, labels=labels)
        score_micro = f1_score(test_y, predictions, 
            labels=labels, average="micro")
        score_macro = f1_score(test_y, predictions, 
            labels=labels, average="macro")

        if verbose:
            print("Fold " + str(n_run) + " : micro_f1=" + str(score_micro) + 
                " macrof1=" + str(score_macro))

        # store for avg
        scores_micro.append(score_micro)
        scores_macro.append(score_macro)
        n_run+=1

        # save the pipeline if better than the current one
        if score_macro > best_f_score :
            best_pipeline = clone(pipeline, True)
            best_f_score = score_macro

    if verbose :
        print("Model Cross Validation complete in %.3f seconds.\n" 
             % (time() - t0))

    scores = {  "mean_score_micro": sum(scores_micro)/len(scores_micro),
                "mean_score_macro": sum(scores_macro)/len(scores_macro),
                "confusion_matrix": confusion,
                "best_macro_score": best_f_score,
                "labels"          : labels}
    
    return best_pipeline, scores
