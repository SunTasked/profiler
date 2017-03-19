def get_features_transformers(verbose=True):
    '''
    Returns a dictionnary containing the features-extractors available 
    '''

    if verbose:
        print("configuring the features transformers ...")

    # transformers
    features_ext = collections.OrderedDict()
    features_ext["Counter-cgram"] = CountVectorizer(analyzer="char")
    features_ext["Counter-1gram"] = CountVectorizer()
    features_ext["Counter-2gram"] = CountVectorizer(ngram_range=(1,2))
    features_ext["Normalizer"   ] = Normalizer(copy=False)
    features_ext["Tf-idf"       ] = TfidfTransformer()
    features_ext["SVDecomposer" ] = TruncatedSVD()

    if verbose:
        print("Available transformers are :")
        for key in features_ext:
            print("   - " + key)
        print()
    
    return features_ext


def get_classifiers(verbose=True):
    '''
    Returns a dictionnary containing the classifiers available 
    '''

    
    if verbose:
        print("configuring the features transformers ...")

    #classifiers
    classifiers = collections.OrderedDict()
    classifiers["Bernouilly NB"     ] = BernoulliNB()
    classifiers["Gaussian NB"       ] = GaussianNB()
    classifiers["Multinomial NB"    ] = MultinomialNB()
    classifiers["Random Forrest"    ] = RandomForestClassifier()
    classifiers["Multi Layered P"   ] = MLPClassifier()
    classifiers["SVM"               ] = LinearSVC()

    if verbose:
        print("Available classifiers are :")
        for key in classifiers:
            print("   - " + key)
        print()

    return classifiers


def dummy_comparator (corpus, labels, output_dir):
    
    # prevent bad specs on output_dir
    if output_dir and output_dir[-1] != "/":
        output_dir = output_dir + "/"

    # get transformers and classifiers
    transformers = get_features_transformers(verbose=True)
    classifiers = get_classifiers(verbose=True)

    # build pipelines
    pipe_chagram = Pipeline([("Counter",transformers["Counter-cgram"])])
    pipe_unigram = Pipeline([("Counter",transformers["Counter-1gram"])])

    pipe_bigram = Pipeline([("Counter",transformers["Counter-2gram"])])

    pipe_tfidf = Pipeline([('Counter', transformers["Counter-2gram"]), 
                           ('Tf-idf', transformers["Tf-idf"])])

    pipe_lsa = Pipeline([('Counter', transformers["Counter-2gram"]),
                         ('Tf-idf', transformers["Tf-idf"]),
                         ('SVD', transformers["SVDecomposer"]),
                         ('Normalizer', transformers["Normalizer"])])

    # prepare treatement : 
    pipelines = []
    pipelines.append(["Chargram",pipe_chagram])
    pipelines.append(["Unigram",pipe_unigram])
    pipelines.append(["Bigram",pipe_bigram])
    pipelines.append(["TF-IDF",pipe_tfidf])
    pipelines.append(["LSA",pipe_lsa])

    # prepare measures tables:
    str_pipelines = [pipe[0] for pipe in pipelines]
    str_classifiers = [clf for clf in classifiers]
    F1_micro = [[0 for x in str_classifiers] for y in str_pipelines]
    F1_macro = [[0 for x in str_classifiers] for y in str_pipelines]
    Time_train = [[0 for x in str_classifiers] for y in str_pipelines]

    
    for pipe in pipelines:
        str_pipe = pipe[0]
        pipeline = pipe[1]

        for str_clf, clf in classifiers.items():
            str_experiment = "FEATURES : " + str_pipe + " - CLASSIFIER " + str_clf
            print(str_experiment)
            time_init = time()

            # add classifier to the pipeline
            pipeline.steps.append((str_clf, clf))
            try:
                model, scores = train_model_cross_validation(corpus, labels, pipeline, verbose)
            except:
                print("some error occured - the features extracted and the classifier are problably incompatible\n")
                pipeline.steps.pop()
                continue
            score_micro = scores["mean_score_micro"]
            score_macro = scores["mean_score_macro"]
            confusion = scores["confusion_matrix"]

            # remove classifier from the pipeline
            pipeline.steps.pop()

            # std output
            print("score micro : ","{0:.3f}".format(score_micro))
            print("score macro : ", "{0:.3f}".format(score_macro))
            print("Confusion Matrix : \n", confusion)
            print()

            # saving micro and macro measures
            i_pipe = str_pipelines.index(str_pipe)
            i_clf = str_classifiers.index(str_clf)
            F1_micro[i_pipe][i_clf] = score_micro
            F1_macro[i_pipe][i_clf] = score_macro
            Time_train[i_pipe][i_clf] = time() - time_init
    
    # final micro and macro measures to save
    export_to_CSV(F1_micro, str_pipelines, str_classifiers, output_dir + "micro.csv")
    export_to_CSV(F1_macro, str_pipelines, str_classifiers, output_dir + "macro.csv")
    export_to_CSV(Time_train, str_pipelines, str_classifiers, output_dir + "time.csv")