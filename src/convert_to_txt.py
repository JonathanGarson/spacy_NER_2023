"""
This code converts all .docx files in a directory to .txt files. It uses the pydocx library which read the xml content beneath a docx file and save the file as a .txt file.
The procedure is faster than in 'convert_to_txt', count 15s for 206 files (0.07s per file).
"""

import os 
import shutil
import glob
from tqdm import tqdm
from docx import Document

ROOT_PATH = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files"

def extract_text_from_docx(file_path, output_file_path):
    """
    Extract text from a docx file and save it in a txt file.

    Args:
        file_path (str): The path to the docx file.
        output_file_path (str): The path to the directory where the txt file will be saved.
    """
    doc = Document(file_path)
    text = []
    
    # Extraction des textes des paragraphes
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    
    # Extraction des textes des tableaux
    # for table in doc.tables:
    #     for row in table.rows:
    #         for cell in row.cells:
    #             text.append(cell.text)
    
    # Ecriture du texte extrait dans un nouveau fichier txt
    with open(output_file_path + "\\" + os.path.basename(file_path) + ".txt", "w", encoding="utf-8") as f:
        for line in text:
            f.write(line + "\n")

def move_files(input_directory, output_directory):
    """
    Move files from one directory to another.

    Args:
        input_directory (str): The path to the directory where the files are.
        output_directory (str): The path to the directory where the files will be moved.
    """
    for file in os.listdir(input_directory):
        if file.endswith(".txt"):
            shutil.move(os.path.join(input_directory, file), output_directory)

# Set your paths
directory = glob.glob(os.path.join(ROOT_PATH, r"data\docx\2022_2023\2022_2023_all\Subset_1\docx\*.docx"))
output_directory = os.path.join(ROOT_PATH, r"data\docx\2022_2023\2022_2023_all\Subset_1\text")

for file in tqdm(directory, desc="Extracting text from docx files", unit="files"):
    if file.endswith(".docx"):
        extract_text_from_docx(file, output_directory)

# Move the files to the output directory
# move_files(os.path.join(ROOT_PATH, r"docx\2022_2023\sample_2023"), output_directory)

print("=========")
print("All done!")