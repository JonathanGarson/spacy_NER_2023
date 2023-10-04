"""
This code will clean the text files from the txt directory. It will just remove the upper case, the \n characters and shorten the name.
"""

import os
import glob
import shutil
from tqdm import tqdm

# Define the ROOT_PATH
ROOT_PATH = r"C:\Users\garsonj\Desktop\Finetuning"

def move_files(input_directory, output_directory):
    for file in os.listdir(input_directory):
        if file.endswith(".txt"):
            shutil.move(os.path.join(input_directory, file), output_directory)

def clean_file(file):
    with open(file, "r", encoding = "utf-8") as f:
        text = f.read()
        text = text.lower()
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

    with open(file, "w", encoding = "utf-8") as f:
        f.write(text)

def shorten_name(file):
    old_name = os.path.basename(file)[0:27]  # Get the file name with extension
    new_name = old_name + ".txt"
    os.rename(file, new_name)
    move_files(os.path.join(ROOT_PATH, output_directory))
        
directory = glob.glob(os.path.join(ROOT_PATH, r"BERT\txt\*.txt"))
output_directory = os.path.join(ROOT_PATH, r"BERT\cleaned_txt")
directory_clean = glob.glob(rf"{output_directory}\*.txt")

for file in tqdm(directory, desc="Cleaning text files", unit="files"):
    clean_file(file)

move_files(r"C:\Users\garsonj\Desktop\Finetuning\BERT\txt", output_directory)

for file in tqdm(directory_clean, desc="Shortening file names", unit="files"):
    shorten_name(file)

print("=========")
print("All done!")

