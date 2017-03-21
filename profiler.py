import sys
from argparse import ArgumentParser

# add path to the python scripts
sys.path.insert(0, './py-scripts')
from utils import integer

########################################
########### Argument Parser ############
########################################

parser = ArgumentParser(description="profiler v1.0")
parser.add_argument("program")
parser.add_argument("action")
parser.add_argument("-l", "--labels",  type=str, dest="selected_labels", 
                    default="",
                    help="specify which labels you wish to use on the output \
                    data (combinations are available) \
                    ['l' for language - 'v' for variety - 'g' for gender]")
parser.add_argument("-c", "--classifier", action='append',
                    dest="selected_classifier", default=[],
                    help="The selected classification algorithm")
parser.add_argument("-f", "--features", dest="selected_features",
                    default=[], action='append',
                    help="The selected set of features")
parser.add_argument("--in","--input-dir", type=str, dest="input_dir",
                    help="specify the directory from which the tweets will be \
                    exctracted")
parser.add_argument("--out","--output-dir", type=str, dest="output_dir",
                    default='./',
                    help="specify the directory in which the result files will \
                    be saved (default is current dir)")
parser.add_argument("--no-cross-validation",  action='store_false',
                    dest="cross_validation", default=True,
                    help="specify if you want to cross validate your model")
parser.add_argument("-s", "--scores",  type=str, dest="scores",
                    default="precision",
                    help="The score function to optimize ('-' separated)")
parser.add_argument("--processed-tweets-dir", type=str,
                    dest="processed_tweets_dir", default='',
                    help="specify the directory from which the tweets will be \
                    exctracted")
parser.add_argument("-v", "--verbosity",  type=integer, dest="verbosity", default=1,
                    help="define the verbosity level that you need \
                    (0 is minimal, 3 is maximal)")
    
args = parser.parse_args(sys.argv)


usr_request = args.action
available_requests = ["train", "classify", "optimize", "compare"]

if usr_request not in available_requests:
    print("ERROR : Unknown user request.")
    print("        Request found : " + usr_request)
    exit()
print(args)
#######################################
########### Program Start #############
#######################################

from tweet_parser import parse_tweets_from_main_dir, parse_tweets_from_dir
from utils import build_corpus

#------------------------------------------------------------------------------
# [Contextual] Compare different algorithms on different features sets
if usr_request == "compare":
    print("--------------------------------------------------------")
    print("Starting classifiers and features extractors comparison.")
    print("--------------------------------------------------------")
    print()

    # Options available :
    #   - input-dir            : input directory for tweet loading
    #   - processed-tweets-dir : output directory for parsed tweet
    #   - verbosity            : verbosity level --> 0 (quiet) to 3 (noisy)
    #   - labels               : which label to train on
    #   - output-dir           : output directory for results persistance
    #   - features             : selected features for extraction
    #   - classifier           : selected classifiers for training or load

    compare_opt = {
        "input-dir"            : args.input_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "verbosity"            : args.verbosity,
        "labels"               : args.selected_labels,
        "output-dir"           : args.output_dir,
        "features"             : args.selected_features,
        "classifier"           : args.selected_classifier
    }

    from act_comparator import compare
    compare(compare_opt)


#------------------------------------------------------------------------------
# [Contextual] Optimize a classification algorithm on a given set of features
elif usr_request == "optimize":
    print("--------------------------")
    print("Starting optimization.")
    print("--------------------------")
    print()

    # TODO


#------------------------------------------------------------------------------
# [Contextual] Train a specified classifier
elif usr_request == "train":
    print("-----------------------------------")
    print("Starting training of a classifier")
    print("-----------------------------------")
    print()

    # Options available :
    #   - input-dir            : input directory for tweet loading
    #   - processed-tweets-dir : output directory for parsed tweet
    #   - verbosity            : verbosity level --> 0 (quiet) to 3 (noisy)
    #   - label                : which label to train on (can be a composition)
    #   - output-dir           : output directory for results persistance
    #   - features             : selected features for extraction
    #   - classifier           : selected classifier for training or load a cfg
    #   - no-cross-validation  : assess if the classifier should be cross-valid

    trainer_opt = {
        "input-dir"            : args.input_dir,
        "processed-tweets-dir" : args.processed_tweets_dir,
        "verbosity"            : args.verbosity,
        "labels"               : args.selected_labels,
        "output-dir"           : args.output_dir,
        "features"             : args.selected_features,
        "classifier"           : args.selected_classifier,
        "cross-validation"     : args.cross_validation
        }

    from act_trainer import train
    train(trainer_opt)

    
#------------------------------------------------------------------------------
# [Contextual] Classify a given test set
elif usr_request == "classify":
    print("-----------------------------------")
    print("Starting classification")
    print("-----------------------------------")
    print()

    # TODO
    