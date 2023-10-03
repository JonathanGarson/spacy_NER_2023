"""
This code create a sample out of the xlsx data downloaded from Qlik Sense platform.
The data are randomly shuffled, divided into subsets of 500 files each and saved in txt files.
"""

import pandas as pd
import os
import random
import shutil

# Set your paths 
data_path = "C:\\Users\\garsonj\\Desktop\\spacy_finetuning\\spacy_files\\data" 
docx_output_path = "C:\\Users\\garsonj\\Desktop\\spacy_finetuning\\spacy_files\\docx"

# Functions
def shuffle_list(list):
    random.shuffle(list)
    return list

def create_subsets(list, start, step_size):
    """
    Create subsets of a list with a given step size.
    
    Args: 
        list (list): The list to be split into subsets.
        start (int): The index of the first item in the list.
        step_size (int): The number of items in each subset.

    Returns:
        subsets (list): A list of subsets.
    """
    subsets = []
    i = start
    while i < len(list):
        end = min(i + step_size, len(list))
        subset = list[i:end]
        subsets.append(subset)
        i += step_size
    return subsets

def save_subsets_to_txt(subsets, path):
    """
    Save subsets to txt files.

    Args:
        subsets (list): A list of subsets.
        path (str): The path to the directory where the txt files will be saved.
    """
    for i, subset in enumerate(subsets, start=1):
        filename = f"{path}/Liste{i}.txt"
        with open(filename, "w") as file:
            for item in subset:
                file.write(str(item) + "\n")

# Read in the data
df = pd.read_excel(f'{data_path}\\training_xlsx\\NAO_2017_2021.xlsx')

# Drop the columns we don't need
df.columns
df_link = df[[
    #'UrlLegifrance', 
    # 'Entreprise', 
    # 'Siret', 
    # 'Secteur', 
    # 'Nature', 
    # 'Titre',
    # 'Naf732', 
    # 'Date Texte', 
    # 'Date Maj', 
    # 'Date Dépot', 
    # 'Date diffusion',
    # 'Date fin', 
    # 'LesSyndicats', 
    # 'LesThemes', 
    # 'type',
    # 'Tranche Effectif(Base siren)', 
    'Fichier', 
    # 'ID'
    ]].copy()

#We make a list of the proapp names
list = df_link['Fichier'].tolist()

shuffled_list = shuffle_list(list)

# Example usage:
my_list = shuffled_list  # Replace this with your list of files
start_index = 0
step_size = 500
subsets = create_subsets(my_list, start_index, step_size)

# Save the subsets to txt files
save_subsets_to_txt(subsets, docx_output_path)

# Create directories and move the corresponding list to them
for i, subset in enumerate(subsets, start=1):
    subset_directory = f"C:\\Users\\garsonj\\Desktop\\spacy_finetuning\\spacy_files\\docx\\2017_2021\\Subset_{i}"
    txt_file = f"C:\\Users\\garsonj\\Desktop\\spacy_finetuning\\spacy_files\\docx\\Liste{i}.txt"
    
    # Remove existing directory and its contents (if it exists)
    if os.path.exists(subset_directory):
        os.system(f"rmdir /s /q {subset_directory}")

    # Create a new directory
    os.mkdir(subset_directory)

    # Move the text file to the new directory
    os.rename(txt_file, os.path.join(subset_directory, f"Liste {i}.txt"))

# Now we make a copy of the Recup_FichiersAccordes.exe in all the directories
for i, subset in enumerate(subsets, start=1):
    subset_directory = f"C:\\Users\\garsonj\\Desktop\\spacy_finetuning\\spacy_files\\docx\\2017_2021\\Subset_{i}"
    shutil.copy(r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\docx\Recup_FichersAccord.exe", subset_directory)

# We rename the Listes in the directories to Liste.txt
# Specify the root directory where your subfolders are located
root_directory = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\docx\2017_2021"

for subdir, _, files in os.walk(root_directory):
    for file in files:
        if file.startswith("List") and file.endswith(".txt"):
            # Construct the source and destination file paths
            source_path = os.path.join(subdir, file)
            destination_path = os.path.join(subdir, "Liste.txt")

            # Rename the file to "Liste.txt"
            os.rename(source_path, destination_path)

root_path = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\data\docx\2017_2021"

# Iterate over subdirectories
for directory in os.listdir(root_path):
    # Check if it's a directory
    if os.path.isdir(os.path.join(root_path, directory)):
        # Create the "text" subfolder within each subdirectory
        text_subfolder = os.path.join(root_path, directory, "text")
        os.makedirs(text_subfolder, exist_ok=True)
        word_subfolder = os.path.join(root_path, directory, "docx")
        os.makedirs(word_subfolder, exist_ok=True)
        print(f"Created 'text' subfolder in '{directory}'")


##############################################################################################################################################################################

# This is optional, it will open the Recup_FichiersAccordes.exe in all the directories which will take few minutes and a lot of storage
# root_directory = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\docx\2017_2021"

# for subdir, _, files in os.walk(root_directory):
#     for file in files:
#         if file.startswith("List") and file.endswith(".txt"):
#             # Construct the source and destination file paths
#             source_path = os.path.join(subdir, file)
#             destination_path = os.path.join(subdir, "Liste.txt")

#             # Rename the file to "Liste.txt"
#             os.rename(source_path, destination_path)