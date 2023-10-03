import spacy
import pandas as pd
import json
from confusion_matrix_spacy_def import automate_confusion_matrix
import os
import sys
sys.path.append(r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\script")

def classify(classifier_model:str, data, label:str):
    """
    This function takes in text stored in csv file and outputs two lists of text for the different labels.
    
    args:
        classifier_model: the path to the model used to classify the text
        data: the path to the csv file containing the text
    """

    # Load the model and data
    nlp_classify = spacy.load(classifier_model)
    data = pd.read_csv(data)

    # Extract the text from the data and class them
    texts = data["text"].tolist()

    for text in texts:
        doc = nlp_classify(text)
        # print(doc.cats,  "-",  text)

    # Store them in corresponding lists
    ppv_list = []
    nppv_list = []

    for text in texts:
        doc = nlp_classify(text)
        if doc.cats.get(label, 0.0) > doc.cats.get(f"N{label}", 0.0):
            ppv_list.append(text)
        else:
            nppv_list.append(text)

    print("la liste ppv:", len(ppv_list)) 
    print(f"la liste nppv:{len(nppv_list)}")
    return ppv_list, nppv_list

def NER(ner_model, list:list):
    """
    This function takes in a list of text and outputs a list of tuples containing the text and the predicted NER labels.

    args:
        ner_model: the path to the model used to predict the NER labels
        list: the list of text to be predicted
    """
    nlp_ner = spacy.load(ner_model)

    # Initialize an empty list to store the output data
    output_data = []

    # Loop through items in the directory
    for text in list:
        doc = nlp_ner(text)
        
        # Extract labeled information
        labeled_data = [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "labels": ent.label_
            }
            for ent in doc.ents
        ]

        # Store text and labeled_data as a tuple and append to the output_data list
        output_data.append((text, {"label": labeled_data}))
    return output_data

def open_json(json_file:str):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_NER_to_json(list:list, output_path:str):
    """
    This function takes in a list of tuples containing the text and the predicted NER labels and outputs a JSON file.

    args:
        list: the list of tuples containing the text and the predicted NER labels
        output_path: the path to the JSON file to store the predicted NER labels
    """
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(list, json_file, ensure_ascii=False, indent=4)
    
def json_file_length(json_file:str):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    lenght = len(data)
    print("Le dossier JSON contient", lenght, "éléments.")

def NER_pipeline(classifier_model, ner_model, data_input, output_path, label):
    """
    This function takes a classifier model, data input, and an output file path.
    It classifies the text using the classifier model and then applies the NER model
    to the classified text, returning a JSON file of predicted NER labels.

    Args:
        classifier_model: The path to the model used to classify the text.
        ner_model: The path to the model used to predict the NER labels.
        data_input: The path to the CSV file containing the text.
        output_path: The path to the JSON file to store the predicted NER labels.
    """
    if os.path.exists(ner_model) & os.path.exists(classifier_model):
        pass
    else:
        print("Le modèle NER n'existe pas.")
        return
        
    # Use the classify function to classify the data
    ppv_list, nppv_list = classify(classifier_model, data_input, label)

    # Use the NER function to predict the NER labels
    # print(f"la liste ppv:{ppv_list}")
    ppv_output = NER(ner_model, ppv_list)
    

    # Save the predicted NER labels to a JSON file
    save_NER_to_json(ppv_output, output_path)

    # Print the length of the JSON file
    json_file_length(output_path)

# Define the paths to the models, data, and output file


def main():
    labels = ['OUV', 'INT', 'CAD', 'NOUV', 'NCAD', 'AG', 'AI', 'TOUS', 'AG OUV', 'AG INT', 'AG CAD', 'AI OUV', 'AI INT', 'AI CAD', 'NOUV AG', 'NCAD AG', 'NOUV AI', 'NCAD AI', 'ATOT',\
            'ATOT OUV', 'ATOT INT', 'ATOT CAD', 'PPV', 'PPVm', 'DATE']

    # Specify the root directory where models and data are located
    root_dir = r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files"

    # Iterate through each label
    for label in labels:
        classifier_model = os.path.join(root_dir, rf'model\classifyer\{label}\model-best')
        ner_model = os.path.join(root_dir, rf'model\unilabel\{label}\model-best')
        data_input = os.path.join(root_dir, r'data\training_csv\data449_cleaned.csv')
        output_path = os.path.join(root_dir, rf'data\predicted_json\labelled_data_{label}_class_to_NER.json')

        # Run NER pipeline for the current label
        NER_pipeline(classifier_model=classifier_model, ner_model=ner_model, data_input=data_input, output_path=output_path, label=label)

        # # Define the target labels and the path to the JSON files
        # target_labels = label
        # true_json_file = os.path.join(root_dir, r'data\training_json\data449.json')
        # pred_json_file = output_path  # Reuse the output path
        # print(true_json_file, pred_json_file, target_labels)

        # # Generate the confusion matrix
        # automate_confusion_matrix(true_json_file, pred_json_file, target_labels, True, os.path.join(root_dir, rf'model\classifyer\{label}\confusion_matrix.png'))
        
if __name__ == "__main__":
    main()