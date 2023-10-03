"""
This code converts all .docx files in a directory to .txt files. It uses the win32com library which open MS word and save the file as a .txt file.
The procedure is fairly slow, count 18mn for 206 files (5.2s per file).
"""


from win32com import client as wc
import glob
import os
import shutil
from tqdm import tqdm

def convert_to_txt(file):
    w = wc.Dispatch('Word.Application')
    doc = w.Documents.Open(file)
    doc.SaveAs(file[:-5] + '.txt', 2)  # Save as a plain text file with .txt extension
    doc.Close()  # Close the Word document
    w.Quit()  # Close the Word application

def move_files(input_directory, output_directory):
    for file in os.listdir(input_directory):
        if file.endswith(".txt"):
            shutil.move(os.path.join(input_directory, file), output_directory)
    
directory = glob.glob(r"C:\Users\garsonj\Desktop\Finetuning\BERT\docx\*.docx")
output_directory = r"C:\Users\garsonj\Desktop\Finetuning\BERT\txt"

for file in tqdm(directory):
    convert_to_txt(file)

move_files(r"C:\Users\garsonj\Desktop\Finetuning\BERT\docx", output_directory)

print("=========")
print("All done!")