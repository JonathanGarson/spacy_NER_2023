import os
import glob
from tqdm import tqdm
import shutil

# Define the ROOT_PATH
ROOT_PATH = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files"

def clean_txt(file):
    """
    Clean the files of their \n, \t, \r, and make them lowercase.
    """
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        text = text.lower()

    # Save the cleaned text back to the file.
    with open(file, 'w', encoding='utf-8') as f:
        f.write(text)

def shorten_filename(file, output_directory):
    """
    Shorten the filename to have just one extension and optionally specify an output directory.
    """
    filename = os.path.basename(file)
    filename_without_extension = filename.split('.')[0] + '.txt'
    output_file = os.path.join(output_directory, filename_without_extension)

    # Use shutil.move to overwrite existing files
    shutil.move(file, output_file)

# FUNCTION VERSION OF THE PYTHON SCRIPT
def clean_auto(input_directory, output_directory):
    files = glob.glob(os.path.join(input_directory, '*.txt'))
    for file in tqdm(files, desc='Cleaning files', unit='files'):
        # Apply the shorten_filename to each file.
        clean_txt(file)
        shorten_filename(file, output_directory)

# TO AUTOMATE THE CLEANING OF THE TXT FILES
# Path to the directory containing the .txt files
# input_directory = os.path.join(ROOT_PATH, r'data\docx\2022_2023\2022_2023_all\Subset_1\text')
# output_directory = os.path.join(ROOT_PATH, r'data\docx\2022_2023\2022_2023_all\Subset_1\text')

# files = glob.glob(os.path.join(input_directory, '*.txt'))

# for file in tqdm(files, desc='Cleaning files', unit='files'):
#     # Apply the shorten_filename to each file.
#     clean_txt(file)
#     shorten_filename(file, output_directory)