# output

Please direct your results and binaries to this directory.\
- trained models
- optimisation results
- comparison results
- classification results
- evaluation results

You should always use sub-directories to avoid versionning your output files

## Post processing directories structure (suggested)

### training models

    dataset-name (pan17)
        |--- language_1
        .       |--- gender
        .       |       \--- result_files
        .       \--- variety
        .               \--- result_files
        .
        |--- language_n
        |       |--- gender
        |       |       \--- result_files
        |       \--- variety
        |               \--- result_files
        |
        \--- languages
                \--- result_files

### optimization files

    optimization-dir
        |--- optimization_1
        .       \--- result-files
        .
        \--- optimization_n
                \--- result-files


### classification author files

    classification-dir
        |--- dataset_1
        .       \--- result-files
        .
        \--- dataset_n
                \--- result-files


### Comparison files

    comparison-dir
        |--- comparison_1
        .       \--- result-files
        .
        \--- comparison_n
                \--- result-files

### Evaluation files

    evaluation-dir
        |--- classification_1
        .       \--- result-files
        .
        \--- classification_n
                \--- result-files

### Notes
The output files of the profiler tool are named useing the following structure:\
<**features-extractor-code-1**>+...+<**features-extractor-code-n**>+<**classifier-code**>_<**file-description**>.ext