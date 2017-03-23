# output

Please direct your results to this directory.

### Directory structure (suggested)

                dataset-name
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

### Notes
The output files of the profiler tool are named useing the following structure:\
<**features-extractor-code-1**>+...+<**features-extractor-code-n**>+<**classifier-code**>_<**file-description**>.ext