# Profiler v1.0
author profiler tool using the scikit-learn library


## Usage :
| OS      | Command Line                                            |
|---------|---------------------------------------------------------|
| Windows | python /path/profiler.py \<command> \<command-options>  |
| Linux   | python3 /path/profiler.py \<command> \<command-options> |

the command and command options are detailed in the following **commands section**

## Commands

Use the folowing commands to conduct your experiments:

| Command      | Purpose                                                          |
|--------------|------------------------------------------------------------------|
| **train**    | train a classifier, validate and persist it                      |
| **classify** | use a trained classifier to predict labels of a given dataset    |
| **compare**  | compare different classifiers on different sets of features      |
| **optimize** | optimize parameters for a specified classifier / set of features |


### train options:

| option                 | mandatory               | purpose                                                                                                                                                                                                                         | Possible Value                                         |
|------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| -l, --labels           | yes                     | Label you wish to predict. Possible labels are : language, variety and gender. Should you choose variety or gender, please specify a directory containing directly all the tweets (no sub directories)  | l, language or v, variety or g, gender                 |
| -c, --classifier       | yes                     | The classifier you wish to use. The available classifiers are specified in the classifiers section. You can also specify a path to a classifier config file.                                                                    | classifier-code                                        |
| -f, --features         | yes                     | The set of features you wish to use. The available set of features are specified in the features section. You can also specify a path to a features extractor config file.                                                      | features-code                                          |
| --in, --input-dir      | yes                     | input directory from which the tweets will be extracted                                                                                                                                                                         | path to the input directory                            |
| --out, output-dir      | no but strongly advised | output directory into which the resulting model and additional training informations will be saved                                                                                                                              | path to the output directory (model and training data) |
| --no-cross-validation  | no                      | Only use if you want to train on the whole dataset without cross-validation. Output directory must be specified in such case.                                                                                                   | /                                                      |
| --processed-tweets-dir | no                      | output directory into which the processed tweets will be stored                                                                                                                                                                 | path to the output directory (tweets)                  |
| -v, --verbosity        | no                      | verbosity level : from 0 (quiet) to 3 (noisy)                                                                                                                                                                                   | [0;3]                                                  |

*Exemple (windows)* : python profiler.py train --labels l --classifier svm --features tfidf --in ./training-data/pan10-03-17 --output-dir ./output/pan17/lang --verbosity 1

### classify options
Not yet implemented

### optimize options
Not yet implemented

### compare options
Not yet implemented

## Classifiers

Below is a list of classifiers currently implemented as default.\
Note v1.0: loading the classifiers with a config file is not implemented yet

| Classifier             | Code | sklearn implementation                                                                                                                                                                                                       | Details                                                                             |
|------------------------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Support Vector Machine | svm  | LinearSVC( C=1.0,  loss='squared_hinge',  penalty='l1',  dual=False,  tol=1e-4,  multi_class='crammer_singer',  fit_intercept=True,  intercept_scaling=1,  class_weight=None,  verbose=0,  random_state=None,  max_iter=500) | svm configuration resulting from preliminary work of optimization on CLEF14 dataset |


## Features

Below is a list of features extractors currently implemented as default.\
Note v1.0: loading the features extractors with a config file is not implemented yet

| Features Extractor                          | code  | sklearn implementation                                                                                                                                                                                                                                                                                                           | Details                                                            |
|---------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Word Count (unigrams + bigrams)             | wc2   | CountVectorizer( input='content', encoding='utf-8', decode_error='ignore', strip_accents=None, analyzer='word', preprocessor=None, tokenizer=None, ngram_range=(1, 2), stop_words=None, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", max_df=1.0, min_df=2, max_features=None, vocabulary=None, binary=False, dtype=np.int64 ) | standard Word Count implementation with both bigrams and unigrams. |
| Text Frequency - Inverse Document Frequency | tfidf | TfidfTransformer( norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)                                                                                                                                                                                                                                                  | This tf-idf implementation relies on a previous wc2 extraction.    |