# Profiler v1.2

author profiler tool using the scikit-learn library


******************************************************************************************************************

## Table of content

* [Requirements](#requirements) -- requirements for this project
* [Usage](#usage) -- how to use the profiler tool
* [Commands](#commands) -- available commands for the profiler tool and the available options
* [Classifiers](#classifiers) -- available classifiers and default implementations
* [Features](#features) -- available features extractors and default implementations
* [Directory Structure](#dir-structure) -- outline regarding the structure of the project directory and sub-directories


******************************************************************************************************************

## Requirements <a name="requirements" />

This tool is coded in Python 3 and has been tested for Python 3.6.

Required additionnal libraries:

| Library      | Version | Notes |
|--------------|---------|-------|
| numpy        | 1.11.X  |       |
| scikit-learn | 0.18.X  |       |
| pandas       | 0.19.X  |       |

**Note**: Some of these libraries require additionnal content to run on your system. Some of which might not be covered by python package managers such as pip. If you don't want to bother with the dependancies I would recommend installing anaconda (3.x) which handles the python scientific environnement for you.

******************************************************************************************************************

## Usage <a name="usage" />
| OS      | Command Line                                            |
|---------|---------------------------------------------------------|
| Windows | python /path/profiler.py \<command> \<command options>  |
| Linux   | python3 /path/profiler.py \<command> \<command options> |

the command and the related options are detailed in the following **commands section**


******************************************************************************************************************

## Commands <a name="commands" />

Use the folowing commands to conduct your experiments:

| Command      | Purpose                                                          |
|--------------|------------------------------------------------------------------|
| **train**    | train a classifier, validate and persist it                      |
| **classify** | use a trained classifier to predict labels of a given dataset    |
| **compare**  | compare different classifiers on different sets of features      |
| **optimize** | optimize parameters for a specified classifier / set of features |
| **evaluate** | evaluate the quality of a classification                         |


### train options

| option                 | mandatory               | purpose                                                                                                                                                                                                                         | Possible Value                                         |
|------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| -l, --labels           | yes                     | Label you wish to predict. Possible labels are : language, variety and gender. Should you choose variety or gender, please specify a directory containing directly all the tweets (no sub directories)  | l, language or v, variety or g, gender                 |
| -c, --classifier       | yes                     | The classifier you wish to use. The available classifiers are specified in the classifiers section. You can also specify a path to a classifier config file.                                                                    | classifier-code                                        |
| -f, --features         | yes                     | The set of features you wish to use. The available set of features are specified in the features section. You can also specify a path to a features extractor config file.                                                      | features-code                                          |
| --in, --input-dir      | yes                     | input directory from which the tweets will be extracted                                                                                                                                                                         | path to the input directory                            |
| --out, output-dir      | yes                     | output directory into which the resulting model and additional training informations will be saved                                                                                                                              | path to the output directory (model and training data) |
| --no-cross-validation  | no                      | Only use if you want to train on the whole dataset without cross-validation. Output directory must be specified in such case.                                                                                                   | /                                                      |
| --processed-tweets-dir | no                      | output directory into which the processed tweets will be stored                                                                                                                                                                 | path to the output directory (tweets)                  |
| -v, --verbosity        | no                      | verbosity level : from 0 (quiet) to 3 (noisy)                                                                                                                                                                                   | [0;3]                                                  |

**Exemple** (windows) : python profiler.py train --labels l --classifier svm --features tfidf --in ./training-data/pan10-03-17 --output-dir ./output/pan17/lang --verbosity 1


### optimize options

| option                 | mandatory               | purpose                                                                                                                                                                                                                         | Possible Value                                         |
|------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| -l, --labels           | yes                     | Label you wish to predict. Possible labels are : language, variety and gender. Should you choose variety or gender, please specify a directory containing directly all the tweets (no sub directories)                          | l, language or v, variety or g, gender                 |
| --hyper-parameters     | yes                     | Path to a file listing the hyper parameters you wish to tune and the values available.                                                                                                                                          | path to the hyper-params file                          |
| --in, --input-dir      | yes                     | input directory from which the tweets will be extracted                                                                                                                                                                         | path to the input directory                            |
| --out, output-dir      | yes                     | output directory into which the resulting model and additional training informations will be saved                                                                                                                              | path to the output directory (model and training data) |
| --no-cross-validation  | no                      | Only use if you want to train on the whole dataset without cross-validation. Output directory must be specified in such case.                                                                                                   | /                                                      |
| --processed-tweets-dir | no                      | output directory into which the processed tweets will be stored                                                                                                                                                                 | path to the output directory (tweets)                  |
| -v, --verbosity        | no                      | verbosity level : from 0 (quiet) to 3 (noisy)                                                                                                                                                                                   | [0;3]                                                  |


### compare options

| option                 | mandatory               | purpose                                                                                                                                                                                                | Possible Value                                         |
|------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| -l, --labels           | yes                     | Label you wish to predict. Possible labels are : language, variety and gender. Should you choose variety or gender, please specify a directory containing directly all the tweets (no sub directories) | l, language or v, variety or g, gender                 |
| -c, --classifier       | yes                     | The classifier you wish to use. The available classifiers are specified in the classifiers section. You can also specify a path to a classifier config file.                                           | classifier-code                                        |
| -f, --features         | yes                     | The set of features you wish to use. The available set of features are specified in the features section. You can also specify a path to a features extractor config file.                             | features-code                                          |
| --in, --input-dir      | yes                     | input directory from which the tweets will be extracted                                                                                                                                                | path to the input directory                            |
| --out, output-dir      | yes                     | output directory into which the resulting model and additional training informations will be saved                                                                                                     | path to the output directory (model and training data) |
| --processed-tweets-dir | no                      | output directory into which the processed tweets will be stored                                                                                                                                        | path to the output directory (tweets)                  |
| -v, --verbosity        | no                      | verbosity level : from 0 (quiet) to 3 (noisy)                                                                                                                                                          | [0;3]                                                  |

**Note**: if you wish to use multiple classifiers and/or multiple sets of features you need to specify each one of them by using the related option (i.e for selecting both svm and mlp for comparison you need to call the *--classifier* option twice)

**Exemple** (windows) : python profiler.py compare --labels l --classifier nbb --classifier svm --features wc2 --features tfidf --in ./training-data/pan10-03-17 --output-dir ./output/pan17/lang --verbosity 1


### classify options
| option            | mandatory               | purpose                                                                                            | Possible Value                                         |
|-------------------|-------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| --classifiers-dir | yes                     | directory containing all the classifiers for each language and label                               | path to the classifiers directory                      |
| --in, --input-dir | yes                     | input directory from which the tweets will be extracted                                            | path to the input directory                            |
| --out, output-dir | yes                     | output directory into which the resulting "predicted-author" files will be saved                   | path to the output directory (model and training data) |
| -v, --verbosity   | no                      | verbosity level : from 0 (quiet) to 3 (noisy)                                                      | [0;3]                                                  |

**Note**: the output files will be saved in the output directory according to the PAN17 specifications


### evaluate options

| option            | mandatory | purpose                                                                       | Possible Value               |
|-------------------|-----------|-------------------------------------------------------------------------------|------------------------------|
| --truth-dir       | yes       | directory containing the truths files (must be dataset-structured alike)      | path to the truth directory  |
| --in, --input-dir | yes       | directory containing all the output files produced by a former classification | path to the input directory  |
| --out, output-dir | yes       | output directory into which the evaluation informations will be saved         | path to the output directory |
| -v, --verbosity   | no        | verbosity level : from 0 (quiet) to 3 (noisy)                                 | [0;3]                        |


******************************************************************************************************************

## Classifiers <a name="classifiers" />

Below is a list of classifiers currently implemented as default.

| Classifier               | Code | sklearn implementation                                                                                                                                                                                                                                                                                                                                                                               | Details                                                                             |
|--------------------------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Naive Bayes (Bernouilly) | nbb  | BernoulliNB( alpha=1.0, binarize=.0, fit_prior=True, class_prior=None)                                                                                                                                                                                                                                                                                                                              | Default sklearn implementation                                                      |
| Support Vector Machine   | svm  | LinearSVC( C=1.0,  loss='squared_hinge',  penalty='l1',  dual=False,  tol=1e-4,  multi_class='crammer_singer',  fit_intercept=True,  intercept_scaling=1,  class_weight=None,  verbose=0,  random_state=None,  max_iter=500)                                                                                                                                                                         | svm configuration resulting from preliminary work of optimization on CLEF14 dataset |
| Multi-Layered Perceptron | mlp  | MLPClassifier( hidden_layer_sizes(100,), activation="relu", solver='adam', alpha=0.0001, batch_size='auto', learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8 ) | Default sklearn implementation                                                      |
| Random Forrest           | rfo  | RandomForestClassifier( n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_split=1e7, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None )                                                                   | Default sklearn implementation                                                      |

**Note** : you can find the configuration files for those classifiers in the config-file directory


******************************************************************************************************************

## Features <a name="features" />

Below is a list of features extractors currently implemented as default.

| Features Extractor                          | code  | sklearn implementation                                                                                                                                                                                                                                                                                                           | Details                                                                                                 |
|---------------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Word Count (unigrams + bigrams)             | wc2   | CountVectorizer( input='content', encoding='utf-8', decode_error='ignore', strip_accents=None, analyzer='word', preprocessor=None, tokenizer=None, ngram_range=(1, 2), stop_words=None, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", max_df=1.0, min_df=2, max_features=None, vocabulary=None, binary=False, dtype=np.int64 ) | standard Word Count implementation with both bigrams and unigrams.                                      |
| Text Frequency - Inverse Document Frequency | tfidf | TfidfTransformer( norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)                                                                                                                                                                                                                                                  | This tf-idf implementation relies on a wc2 extraction.                                                  |
| Latent Semantic Analysis                    | lsa   | TruncatedSVD(  n_components=1000, algorithm="randomized", n_iter=10, random_state=42, tol=0. )                                                                                                                                                                                                                                   | This LSA implementation (which is basically a singular value decomposition relies on a tfidf extraction |

**Note** : you can find the configuration files for those features extractors in the config-file directory


******************************************************************************************************************

## directory structures <a name="dir-structure" />

The project directory is structured as follow : 

    profiler-project-directory
        |--- config-files           // all the default and custom config files for the profiler tools
        |--- output                 // all the output files of the profiler tool execution
        |--- py-scripts             // the python scripts used by the profiler
        |--- trained-classifiers    // the trained classifiers available for the classification functionnality
        |--- training-data          // all datasets available for training
        |--- .gitignore             // gitignore file
        |--- profiler.py            // main python script
        \--- README.md              // this file

**Note** : The roles of the different directories are explained with more details in the README.md files they contain.