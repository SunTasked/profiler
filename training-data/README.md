# training data

Please save your datasets in this directory 
The datasets shall respect the PAN'17 specifications.

### dataset directory structure
            main_dir
                |--- language_1_dir         // directory containing all the tweets for the language_1 
                .       |--- author_1.xml   // file containing the tweets from author 1
                .       .--- ...
                .       |--- author_n.xml   // file containing the tweets from author n
                .       \--- truth.txt      // file containing all the authors' labels
                .
                \--- language_n_dir         // directory containing all the tweets for the language_n 
                        |--- author_1.xml   // file containing the tweets from author 1
                        .--- ...
                        |--- author_n.xml   // file containing the tweets from author n
                        \--- truth.txt      // file containing all the authors' labels

### Note
The datasets are not persisted within the git structure.\
One dataset is available here: https://goo.gl/oyWKvM
