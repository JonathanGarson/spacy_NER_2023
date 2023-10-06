import spacy
import pandas as pd
import json
from confusion_matrix_spacy_def import automate_confusion_matrix
import os

def classify(classifier_model: str, data_path: str, label: str):
    """
    This function takes in text stored in a CSV file and outputs two lists of text for the different labels.
    
    Args:
        classifier_model: the path to the model used to classify the text
        data_path: the path to the CSV file containing the text
        label: the label you want to use for classification
    """
    
    # Load the model
    nlp_classify = spacy.load(classifier_model)

    # Read the data from the CSV file
    data = pd.read_csv(data_path)

    # Initialize lists to store positive and negative texts
    pos_list = []
    neg_list = []

    # Iterate through each text and classify it
    for text in data["text"]:
        doc = nlp_classify(text)
        # print(doc.cats)  # Print the classification scores
        if doc.cats.get(label, 0.0) > doc.cats.get("0", 0.0):
            # print("Assigned to pos_list")
            pos_list.append(text)
        else:
            # print("Assigned to neg_list")
            neg_list.append(text)


    print("la liste pos_list:", len(pos_list)) 
    print(f"la liste neg_list:{len(neg_list)}")
    
    return pos_list, neg_list

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
    length = len(data)
    print(length, "éléments ont été analysés.")

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
    # Use the classify function to classify the data
    pos_list, neg_list = classify(classifier_model, data_input, label)

    # Use the NER function to predict the NER labels
    # print(f"la liste ppv:{pos_list}")
    ppv_output = NER(ner_model, pos_list)
    

    # Save the predicted NER labels to a JSON file
    save_NER_to_json(ppv_output, output_path)

    # Print the length of the JSON file
    json_file_length(output_path)

def main():
    labels = ['OUV', 'INT', 'CAD', 'NOUV', 'NCAD', 'AG', 'AI', 'TOUS', 'AG OUV', 'AG INT', 'AG CAD', 'AI OUV', 'AI INT', 'AI CAD', 'NOUV AG', 'NCAD AG', 'NOUV AI', 'NCAD AI', 'ATOT',\
            'ATOT OUV', 'ATOT INT', 'ATOT CAD', 'PPV', 'PPVm', 'DATE']


    # Iterate through each label
    for label in labels:
        print("=====================================================")
        print(f"Running NER pipeline for {label}...")
        print("=====================================================")

        # Define the paths to the models, data, and output
        classifier_model = rf"./models/classifier/{label}/model-best"
        ner_model = rf"./models/ner/unilabel/{label}/model-best"
        data_input = r'./data/processed/data449_text.csv'
        output_path = rf"./data/processed/predicted_labelled_{label}.json"
        output_text = rf"./reports/predicted_labelled_{label}_results.txt"

        if os.path.exists(ner_model) and os.path.exists(classifier_model):
            pass
        else:
            print("Le modèle NER n'existe pas.")
            print("classeur:", classifier_model, "ner: ", ner_model)
            continue
        
        # Run NER pipeline for the current label
        NER_pipeline(classifier_model=classifier_model, ner_model=ner_model, data_input=data_input, output_path=output_path, label=label)

        # Define the target labels and the path to the JSON files
        target_labels = label
        true_json_file = r'.\data\raw\data449.json'
        pred_json_file = output_path  # Reuse the output path
        print(true_json_file, pred_json_file, target_labels)

        # Generate the confusion matrix
        automate_confusion_matrix(true_json_file, pred_json_file, target_labels, True, rf'.\reports\figures\confusion_matrix_{label}.png', output_text=output_text)
        print(f"saved confusion matrix to : .\reports\figures\confusion_matrix_{label}.png")

if __name__ == "__main__":
    main()