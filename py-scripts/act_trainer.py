from time import time

from numpy import array
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

from act_classifier import predict_author_proba
from classifiers import get_classifier
from features import get_features_extr
from persistance import save_model, save_scores
from dataset_parser import parse_tweets_from_main_dir, parse_tweets_from_dir
from pipeline import get_pipeline
from utils import build_corpus, abort_clean, print_scores, get_printable_tweet
from utils import get_classifier_name, get_features_extr_name, get_labels


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
    if not (options["label_type"]):
        abort_clean("Labels not specified", "expected 'l', 'g' or 'v'")
    
    if not (options["features"]):
        abort_clean("Features not specified")

    if not (options["classifier"]):
        abort_clean("Classifier not specified")


    #--------------------------------------------------------------------------
    # Load the tweets in one language for variety or gender classification
    Authors = parse_tweets_from_dir(
        input_dir=options["input-dir"], 
        output_dir=options["processed-tweets-dir"],
        label=True,
        strategy=options["strategy"],
        verbosity_level=options["verbosity"])

    if not (Authors):
        abort_clean("Tweets loading failed")


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
            print("Model Training with cross validation\n")

        pipeline, scores = train_model_cross_validation(
                            authors=Authors,
                            label_type = options["label_type"], 
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
            print("Model Training without cross validation\n")
        if not(options["output-dir"]):
            abort_clean(
                "No output directory specified.", 
                "Training without persisting is not allowed")
        pipeline = train_model(
                    authors=Authors,
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
    train_tweets = corpus['tweets']
    train_labels = corpus['labels']

    # train the pipeline
    pipeline.fit(train_tweets, train_labels)
    
    if verbose :
        print("Model training complete in %.3f seconds\n"  % (time() - t0))

    return pipeline


def train_model_cross_validation(authors, label_type, pipeline, tweet_level=False, verbose=1):
    '''
    Takes a pipeline and train it on the specified corpus.
    Processes a cross-validation algorithm (K-fold) in order to evaluate the
    quality of the model.
    Returns the best trained pipeline (in terms of macro f-score).
    '''

    labels = get_labels(
        lang=authors[0]["lang"],
        label_type=label_type )
        
    if not(labels):
        abort_clean("Could not extract labels")
    if verbose :
        print("Labels extraction succeded.")
        print("Available labels : " + " / ".join(labels) + "\n")
    

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
    authors = array(authors)
    for train_indices, test_indices in k_fold.split(authors):
        
        # build train corpus
        train_authors = authors[train_indices]
        train_corpus = build_corpus(
            authors=train_authors,
            label_type=label_type,
            verbosity=verbose)
            
        # build test corpus
        test_authors = authors[train_indices]

        # train model
        pipeline = train_model(
            corpus=train_corpus, 
            pipeline=pipeline,
             verbose=0)

        # test model
        truthes = []
        predictions = []
        for author in test_authors : 
            var_classes, var_predictions = predict_author_proba(
                author=author,
                model=pipeline )
            var_max_idx = var_predictions.index(max(var_predictions))
            label_predicted = var_classes[var_max_idx]
            predictions.append(label_predicted)
            truthes.append(author[label_type])
            
        
        # compute metrics
        confusion += confusion_matrix(truthes, predictions, labels=labels)
        score_micro = f1_score(truthes, predictions, 
            labels=labels, average="micro")
        score_macro = f1_score(truthes, predictions, 
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
