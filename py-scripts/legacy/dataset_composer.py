#!/usr/bin/python3

from argparse import ArgumentParser
from shutil import copyfile
import sys

sys.path.insert(0, '../')
from utils import format_dir_name, get_language_dir_names, create_dir

parser = ArgumentParser(description="profiler v1.0")
parser.add_argument("program")
parser.add_argument("--in","--input-dir", type=str, dest="input_dir",
                    help="specify the input directory")
parser.add_argument("--out","--output-dir", type=str, dest="output_dir",
                    help="specify the output directory")

args = parser.parse_args(sys.argv)

input_dir = format_dir_name(args.input_dir)
output_dir = format_dir_name(args.output_dir)
lang_dir_name = get_language_dir_names()

for lang in lang_dir_name:
    input_lang_dir = format_dir_name(input_dir + lang)
    output_lang_dir = format_dir_name(output_dir + lang)
    output_male_dir = format_dir_name(output_lang_dir + "male")
    output_female_dir = format_dir_name(output_lang_dir + "female")

    truth_file = open(input_lang_dir + "truth.txt")
    truth_lines = [x.strip().split(':::') for x in truth_file.readlines()]
    truth_file.close()

    create_dir(output_male_dir)
    copyfile(input_lang_dir + "truth.txt", output_male_dir + "truth.txt")
    create_dir(output_female_dir)
    copyfile(input_lang_dir + "truth.txt", output_female_dir + "truth.txt")

    for line in truth_lines:
        input_file = input_lang_dir
        if line[1] == "male":
            copyfile(input_lang_dir + line[0] + ".xml", 
                     output_male_dir + line[0] + ".xml")
        else:
            copyfile(input_lang_dir + line[0] + ".xml", 
                     output_female_dir + line[0] + ".xml")