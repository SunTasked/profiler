from time import time

from act_trainer import train_model_cross_validation
from classifiers import get_classifier
from features import get_features_extr
from persistance import save_scores, save_comparison_table
from dataset_parser import parse_tweets_from_main_dir, parse_tweets_from_dir
from pipeline import get_pipeline
from utils import build_corpus, abort_clean, print_scores
from utils import create_dir, get_classifier_name, get_features_extr_name

#------------------------------------------------------------------------------
#----------------------------- COMPARISON MODULE ------------------------------
#------------------------------------------------------------------------------

def compare(options):
    '''
    Compare a set of specified classifiers on a specified dataset using 
    specified features
    Will proceed as follows :
        - loads the dataset
        - builds the corpus
        - loads the classifiers
        - loads the features extractors
        - builds the execution pipelines
        - trains the different classifiers on the corpus
        - saves the scores obtained by each classifier on each set of features
    '''

    #--------------------------------------------------------------------------
    # Check basic requirements
    if not (options["label_type"]):
        abort_clean("label type not specified", "expected 'l', 'g' or 'v'")
    
    if not (options["features"]):
        abort_clean("Features not specified")

    if not (options["classifier"]):
        abort_clean("Classifier not specified")


    #--------------------------------------------------------------------------
    # Load the tweets
    Authors = parse_tweets_from_dir(
        input_dir=options["input-dir"], 
        output_dir=options["processed-tweets-dir"],
        label=True,
        aggregation=options["aggregation"],
        verbosity_level=options["verbosity"])

    if not (Authors):
        abort_clean("Tweets loading failed")


    #--------------------------------------------------------------------------
    # Load the classifiers
    
    classifier_str_list = []
    if isinstance(options["classifier"], list):
        classifier_str_list = options["classifier"]
    else :
        classifier_str_list = [options["classifier"]]

    classifiers = [ get_classifier(
                        classifier_str=clf,
                        config=None,
                        verbose=False) for clf in classifier_str_list]

    if options["verbosity"]:
        print("Classifiers Loaded: ")
        for clf in classifiers:
            print("    - '" + clf[0] + "'")
        print()

    #--------------------------------------------------------------------------
    # Load the features extractors

    extractors_str_list = options["features"]

    extractors = [
        get_features_extr(
            features_str_list=extr,
            verbose=False ) 
        for extr in extractors_str_list]

    if options["verbosity"]:
        print("Features extractors Loaded: ")
        for extrs in extractors:
            print("    - '" + extrs[0] + "'")
        print()


    #--------------------------------------------------------------------------
    # Prepare results informations supports

    F1_micro = [[0 for x in classifiers] for y in extractors]
    F1_macro = [[0 for x in classifiers] for y in extractors]
    Time_train = [[0 for x in classifiers] for y in extractors]

    output_dir = options["output-dir"]
    individual_scores_dir = output_dir + "indiv_scores/"
    create_dir(individual_scores_dir)


    #--------------------------------------------------------------------------
    # Start the model comparison

    t0 = time()
    total_iteration = len(classifiers)*len(extractors)
    if options["verbosity"]:
        print("Starting model comparisons")

    # Loop for each pair features-extractor/classifier
    for idx_extr, extr in enumerate(extractors):
        extr_name = get_features_extr_name(extr)

        for idx_clf, clf in enumerate(classifiers):
            clf_name = get_classifier_name(clf)

            if options["verbosity"]:
                iteration_number = (idx_extr)*len(classifiers) + idx_clf + 1
                print("Iteration : " + str(iteration_number) +
                    "/" + str(total_iteration))
                print("Testing : Features: " + extr_name + 
                    " | Classifier: " + clf_name)
            
            t0_step = time()

            # Build pipeline
            pipeline = get_pipeline(
                features_extr=extr,
                classifier=clf,
                verbose=False )

            # Start training + cross validation
            try:
                model, step_scores = train_model_cross_validation(
                        authors=Authors,
                        label_type = options["label_type"], 
                        pipeline=pipeline,
                        verbose=False)
            except:
                print("some error occured - the features extracted and the \
                    classifier are problably incompatible\n")
                continue
            
            if options["verbosity"]:
                print("Training complete in " + str(round(time() - t0_step)) +
                     " seconds")
                print_scores(step_scores)
                print()
            
            # Save scores
            save_scores(
                scores=step_scores, 
                output_dir=individual_scores_dir, 
                filename=extr_name+"+"+clf_name, 
                verbose=False )
            F1_micro[idx_extr][idx_clf] = step_scores["mean_score_micro"]
            F1_macro[idx_extr][idx_clf] = step_scores["mean_score_macro"]
            Time_train[idx_extr][idx_clf] = round(time() - t0_step)
    
    # Save final micro and macro measuresand execution time
    save_comparison_table(F1_micro, extractors, classifiers, output_dir +
        "micro.csv")
    save_comparison_table(F1_macro, extractors, classifiers, output_dir +
        "macro.csv")
    save_comparison_table(Time_train, extractors, classifiers, output_dir +
        "time.csv")

    if options["verbosity"]:
        print("Comparison task complete in " + str(round(time()-t0)) + " s")