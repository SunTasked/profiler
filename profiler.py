import sys
from argparse import ArgumentParser

# add path to the python scripts
sys.path.insert(0, './py-scripts')
from utils import integer, abort_clean, dir_exists, clean_options

########################################
########### Argument Parser ############
########################################

parser = ArgumentParser(description="profiler v1.0")
parser.add_argument("program")
parser.add_argument("action")
parser.add_argument("-l", "--label-type",  type=str, dest="label_type", 
                    default="",
                    help="specify which type of labels you wish to use on the\
                    output data (combinations are available) \
                    ['v' for variety - 'g' for gender]")
parser.add_argument("-s", "--strategy",  type=str, dest="strategy", 
                    default="dissociate",
                    help="specify if you want to : \
                    \"aggregate\" the tweets in one chunk or \
                    \"dissociate\" the tweets (keep them as they are).")
parser.add_argument("-c", "--classifier", action='append',
                    dest="classifier", default=[],
                    help="The selected classification algorithm")
parser.add_argument("-f", "--features", dest="features",
                    default=[], action='append',
                    help="The selected set of features")
parser.add_argument("--in","--input-dir", type=str, dest="input_dir",
                    help="specify the directory from which the tweets will be \
                    exctracted")
parser.add_argument("--out","--output-dir", type=str, dest="output_dir",
                    help="specify the directory in which the output files \
                    will be saved")
parser.add_argument("--classifiers-dir", type=str, dest="classifiers_dir",
                    help="specify the directory from which the execution \
                    pipelines will be loaded")
parser.add_argument("--classification-type", type=str, dest="classification_type",
                    help="specify the type of classification required \
                    [loose / successive]")
parser.add_argument("--hyper-parameters", type=str, dest="hyper_parameters",
                    default="",
                    help="specify a config file listing the hyper parameters \
                    to be tuned")
parser.add_argument("--no-cross-validation", action='store_false',
                    dest="cross_validation", default=True,
                    help="specify if you want to cross validate your model")
parser.add_argument("--truth-dir", type=str, dest="truth_dir",
                    help="specify a truth directory to evaluate the classification \
                    results")
parser.add_argument("--scores",  type=str, dest="scores",
                    default="precision",
                    help="The score function to optimize ('-' separated)")
parser.add_argument("--processed-tweets-dir", type=str,
                    dest="processed_tweets_dir", default='',
                    help="specify the directory from which the tweets will be \
                    exctracted")
parser.add_argument("-v", "--verbosity",  type=integer, dest="verbosity", 
                    default=1,
                    help="define the verbosity level that you need \
                    (0 is minimal, 3 is maximal)")
    

args = parser.parse_args(sys.argv)
usr_request = args.action

    
#######################################
########### Program Start #############
#######################################


# Options available for every command:
#   - output-dir           : output directory for results persistance
#   - processed-tweets-dir : output directory for parsed tweet
#   - verbosity            : verbosity level --> 0 (quiet) to 3 (noisy)


# Check and Clean arguments :
args = clean_options(args)

#------------------------------------------------------------------------------
# [Contextual] Classify a given dataset
if usr_request == "classify":
    print("-----------------------------------")
    print("Starting classification")
    print("-----------------------------------")
    print()

    # Options available :
    #   - execution-dir        : path to a folder containing the pipe binaries
    #   - input-dir            : input directory for tweet loading
    #   - output-dir           : output directory for resulting author files

    classifier_opt = {
        "classifiers-dir"      : args.classifiers_dir,
        "classification-type"  : args.classification_type,
        "input-dir"            : args.input_dir,
        "output-dir"           : args.output_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "verbosity"            : args.verbosity
        }

    from act_classifier import classify
    classify(classifier_opt)


#------------------------------------------------------------------------------
# [Contextual] Compare different algorithms on different features sets
elif usr_request == "compare":
    print("--------------------------------------------------------")
    print("Starting classifiers and features extractors comparison.")
    print("--------------------------------------------------------")
    print()

    # Options available :
    #   - classifier           : classifiers code / path to config file
    #   - features             : features extractors code / path to config file
    #   - input-dir            : input directory for tweet loading
    #   - labels               : which labels to train on

    compare_opt = {
        "classifier"           : args.classifier,
        "features"             : args.features,
        "input-dir"            : args.input_dir,
        "labels"               : args.label_type,
        "output-dir"           : args.output_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "verbosity"            : args.verbosity
    }

    from act_comparator import compare
    compare(compare_opt)


#------------------------------------------------------------------------------
# [Contextual] Train a specified classifier
elif usr_request == "evaluate":
    print("-----------------------------------")
    print("Starting Evaluating results")
    print("-----------------------------------")
    print()

    # Options available :
    #   - input-dir            : input directory for predicted author files

    evaluator_opt = {
        "truth-dir"            : args.truth_dir,
        "input-dir"            : args.input_dir,
        "output-dir"           : args.output_dir,
        "verbosity"            : args.verbosity
        }

    from act_evaluator import evaluate
    evaluate(evaluator_opt)



#------------------------------------------------------------------------------
# [Contextual] Optimize a classification algorithm on a given set of features
elif usr_request == "optimize":
    print("--------------------------")
    print("Starting optimization.")
    print("--------------------------")
    print()

    # Options available :
    #   - hyper-params         : a path to a file listing the hyper parameters
    #                            to be tuned (name + values)
    #   - input-dir            : input directory for tweet loading
    #   - labels               : which labels to train on

    optimize_opt = {
        "hyper-parameters"     : args.hyper_parameters,
        "input-dir"            : args.input_dir,
        "labels"               : args.label_type,
        "output-dir"           : args.output_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "verbosity"            : args.verbosity
    }

    from act_optimizer import optimize
    optimize(optimize_opt)


#------------------------------------------------------------------------------
# [Contextual] Train a specified classifier
elif usr_request == "train":
    print("-----------------------------------")
    print("Starting training of a classifier")
    print("-----------------------------------")
    print()

    # Options available :
    #   - classifier           : classifiers code / path to config file
    #   - features             : features extractors code / path to config file
    #   - input-dir            : input directory for tweet loading
    #   - labels               : which labels to train on
    #   - no-cross-validation  : assess if the classifier should be cross-valid

    trainer_opt = {
        "classifier"           : args.classifier,
        "cross-validation"     : args.cross_validation,
        "features"             : args.features,
        "input-dir"            : args.input_dir,
        "label_type"           : args.label_type,
        "output-dir"           : args.output_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "strategy"             : args.strategy,
        "verbosity"            : args.verbosity
        }

    from act_trainer import train
    train(trainer_opt)


   
#------------------------------------------------------------------------------
# [Contextual] Unknown Request
else:
    abort_clean("ERROR : Unknown user request.",
        "Request found : " + usr_request)
