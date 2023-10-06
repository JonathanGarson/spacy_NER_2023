import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
   
def import_label_studio_data(filename):
    """
    This function imports the data from Label Studio JSON file and returns the data in the format required for training.
    It also allows selecting specific labels to train the model on with the "label" argument.

    Args:
        filename (str): The path to the JSON file.
        label (list): The list of labels to train the model on.

    Returns:
        A list of tuples containing (text, {"entities": entities}).
    """
    TRAIN_DATA = []
    with open(filename,'rb') as fp:
        training_data = json.load(fp)
    for text in training_data:
        entities = []
        info = text.get('text')
        entities = []
        if text.get('label') is not None:
            list_ = []
            for label in text.get('label'):
                list_.append([label.get('start'), label.get('end')])
            a = np.array(list_)
            overlap_ind =[]
            for i in range(0,len(a[:,0])):
                a_comp = a[i]
                x = np.delete(a, (i), axis=0)
                overlap_flag = any([a_comp[0] in range(j[0], j[1]+1) for j in x])
                if overlap_flag:
                    overlap_ind.append(i)
                    
            for ind, label in enumerate(text.get('label')):
                if ind in overlap_ind:
                    iop=0
                else:
                    if label.get('labels') is not None:
                        entities.append((label.get('start'), label.get('end') ,label.get('labels')[0]))
        TRAIN_DATA.append((info, {"entities" : entities}))
    return TRAIN_DATA


def spacy_to_dataframe(data):
    """
    This function takes the data in the format returned by the import_label_studio_data function and returns a pandas dataframe of two columns: text and label.

    Args:
        data: The data in the format returned by the import_label_studio_data function.

    Returns:
        A pandas dataframe of two columns: text and label.
    """
    text_data = [text for text, _ in data]
    labels = [label for _, label in data]

    df = pd.DataFrame({'text': text_data, 'label': labels})
    return df

def dummy_label_true(df, label):
    """
    This function creates a dummy variable for the target label.

    Args:
        df (DataFrame): The DataFrame containing the text and label columns.
    """
    # Create a new column called "label_dummy" and initialize with zeros
    df["label_dummy"] = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        labels = row["label"]["entities"]  # Access the entities list in the tuple
        for label in labels:
            target = label[2]
            if target == label:
                df.at[index, "label_dummy"] = 1  # Set the value to 1 for the current row

    # Print the DataFrame to verify the changes
    # print(df["label_dummy"].value_counts())
    return df

def dummy_label(df,label, pred=False):
    """
    This function creates a dummy variable for the target label.

    Args:
        df (DataFrame): The DataFrame containing the text and label columns.
    """
    # Create a new column called "label_dummy" and initialize with zeros
    df["label_dummy"] = 0

    start = 0 

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # print(index)
        labels = row["label"]["entities"]  if not pred else row["label"]["label"]
        if pred:
           if label in set(map(lambda d:d["labels"],labels)):
            df.at[index, "label_dummy"] = 1
        else :
            if label in set(map(lambda x:x[2],labels)):
                df.at[index, "label_dummy"] = 1  # Set the value to 1 for the current row
                start +=1
    print(start)
    return df

def clean_dataset(data):
    """
    This function cleans the dataset by removing rows with missing values and dropping the "label" column.
    It also renames the "label_dummy" column to "label".

    Args:
        data (DataFrame): The DataFrame containing the text, label, and label_dummy columns.
    """
    data.dropna(axis=0, how='any', inplace=True)
    # Now we can drop the "label" column and rename the "label_dummy" column to "label"
    if 'label_dummy' in data.columns:
        data.drop("label", axis=1, inplace=True)
        data.rename(columns={"label_dummy": "label"}, inplace=True)
    else:
        pass
    # print(data.head())
    return data

def dummy_label_pred(df_pred):
    """
    This function creates a dummy variable for the target label.

    Args:
        df_pred (DataFrame): The DataFrame containing the text and label columns.
    """
    # Create a new column called "label_dummy" and initialize with zeros
    df_pred["label_dummy"] = 0

    for index, row in df_pred.iterrows():
        label_data = row['label']
        if label_data == {'label': []}:
            df_pred.at[index, 'label_dummy'] = 0
        else:
            df_pred.at[index, 'label_dummy'] = 1

    # Print the DataFrame to verify the changes
    # print(df_pred["label_dummy"].value_counts())
    return df_pred


def prepare_for_confusion_matrix(df_true, df_pred):
    """
    This function merges the true and predicted labels into a single dataframe.

    Args:
        df_true (DataFrame): The DataFrame containing the text and label columns.
        df_pred (DataFrame): The DataFrame containing the text and label columns.
    """
    df_whole = pd.merge(left=df_true, right=df_pred, on='text', how="inner")
    # print(df_whole.shape, df_true.shape, df_pred.shape)
    true_label_data = df_whole['label_x'].tolist()
    pred_label_data = df_whole['label_y'].tolist()
    # print(true_label_data, pred_label_data)
    return true_label_data, pred_label_data


def spacy_confusion_matrix(true_label_data, pred_label_data, label, save_confusion_matrix=False, output_path=None):
    """
    This function plots the confusion matrix for the predictions made by the spaCy model.
    """
    confusion = confusion_matrix(true_label_data, pred_label_data)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size if needed
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['0', label], yticklabels=['0', label])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if save_confusion_matrix and output_path:
        plt.savefig(output_path)

    plt.show()

def print_statistics(true_label_data, pred_label_data):
    """
    This function prints the precision, recall, and F1 score for the predictions made by the spaCy model.
    
    Args:
        true_label_data (list): The list of true labels.
        pred_label_data (list): The list of predicted labels.
    
    """
    confusion = confusion_matrix(true_label_data, pred_label_data)
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    TP = confusion[1, 1]

    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    return precision, recall, f1_score

def write_statistics_to_file(precision, recall, f1_score, output_path:str):
    """
    This function write down in files the three statistics: precision, recall, and F1 score.

    Args:
        precision (float): The precision score.
        recall (float): The recall score.
        f1_score (float): The F1 score.
        output_path (str): The path to the output file.
    """
    with open(output_path, "w") as f:
        f.write("Precision: " + str(precision) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("F1 Score: " + str(f1_score) + "\n")


def automate_confusion_matrix(true_json, pred_json, label, save_confusion_matrix=False, output_path=None, output_text=None):
    # for the true json
    true_data = import_label_studio_data(true_json)
    df_true = spacy_to_dataframe(true_data)
    df_true = dummy_label(df_true, label=label)
    df_true = clean_dataset(df_true)

    # for the predicted json
    with open(pred_json, "r", encoding="utf-8") as f:
        data_pred = json.load(f)

    df_pred = spacy_to_dataframe(data_pred)
    df_pred = dummy_label(df_pred, label=label, pred=True)
    df_pred = clean_dataset(df_pred)

    # merge the true and predicted labels into a single dataframe
    true_label_data, pred_label_data = prepare_for_confusion_matrix(df_true, df_pred)

    # plot the confusion matrix
    spacy_confusion_matrix(true_label_data, pred_label_data, label, save_confusion_matrix=save_confusion_matrix, output_path=output_path)

    # print the statistics
    precision, recall, f1_score = print_statistics(true_label_data, pred_label_data)

    # write the statistics to a file
    write_statistics_to_file(precision, recall, f1_score, output_path=output_text)