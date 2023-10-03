import json
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin
import os
import subprocess

######### functions #########

import json
import numpy as np

# https://www.kaggle.com/code/kiruthigaa/ner-model-train-test-using-spacy-label-studio

def import_label_studio_data(filename):
    """
    This function imports the data from label-studio and converts it into the format required by spacy.
    """
    TRAIN_DATA = []
    
    with open(filename,'rb') as fp:
        training_data = json.load(fp)
    for text in training_data:
        entities = []
        info = text.get('text')
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

def dummy_label(df, target_label):
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
            if target == f"{target_label}":
                df.at[index, "label_dummy"] = 1  # Set the value to 1 for the current row

    # Print the DataFrame to verify the changes
    print(df["label_dummy"].value_counts())
    return df

def clean_dataset(data):
    """
    This function cleans the dataset by removing rows with missing values and dropping the "label" column.
    It also renames the "label_dummy" column to "label".

    Args:
        data (DataFrame): The DataFrame containing the text, label and label_dummy columns.
    """
    data.dropna(axis=0, how='any', inplace=True)
    # Now we can drop the "label" column and rename the "label_dummy" column to "label"
    if 'label_dummy' in data.columns:
        data.drop("label", axis=1, inplace=True)
        data.rename(columns={"label_dummy": "label"}, inplace=True)
    else:
        pass
    print(data.head())
    return data

def create_tuples(data):
    """
    This function creates a list of tuples from the DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the text and label columns.
    """
    data = list(data[["text", "label"]].sample(frac=1).itertuples(index=False, name=None))
    return data

def dataframe_to_tuples(df):
    """
    Converts a DataFrame with two columns into a list of tuples.
    
    Args:
        df (pd.DataFrame): The input DataFrame with two columns.
        
    Returns:
        list of tuples: A list of tuples where each tuple contains values from the two columns.
    """
    tuples = [(row['text'], row['label']) for _, row in df.iterrows()]
    return tuples

def split_data(data, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, random_seed=None):
    """
    Split a dataset into training, validation, and test sets.

    Parameters:
    - data: The dataset to be split.
    - train_ratio: The ratio of data to be allocated to the training set (default: 0.75).
    - val_ratio: The ratio of data to be allocated to the validation set (default: 0.15).
    - test_ratio: The ratio of data to be allocated to the test set (default: 0.10).
    - random_seed: Seed for the random shuffling (default: None, which results in non-reproducible shuffling).

    Returns:
    - A tuple containing three sets: (train_set, val_set, test_set)
    """
    # Calculate the total size of the dataset
    total_size = len(data)
    
    # Calculate the sizes of each split
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Set the random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Shuffle the data
    shuffled_data = np.random.permutation(data)
    shuffled_data =list(map(lambda x:(str(x[0]),int(x[1])),shuffled_data))
   
    # Split the data into three sets
    train_set = shuffled_data[:train_size]
    val_set = shuffled_data[train_size:train_size + val_size]
    test_set = shuffled_data[train_size + val_size:]
    
    # Print the size of each set
    print("Training set size:", len(train_set))
    print("Validation set size:", len(val_set))
    print("Test set size:", len(test_set))

    return train_set, val_set, test_set

def convert(data, outfile, pos_label, neg_label="0"):
    """
    This function converts the data into spaCy's binary format and saves it to the specified file.

    Args:
        data: The data to be converted.
        outfile: The file to which the converted data should be saved.
    """
    db = spacy.tokens.DocBin()

    for doc, label in nlp.pipe(data, as_tuples=True):

        doc.cats[f"{pos_label}"] = label == 0
        doc.cats[f"{neg_label}"] = label == 1
     
        db.add(doc)
    
    db.to_disk(outfile)
    print("Data saved to:", outfile)

def create_folder_and_init(name):
    folder_name = rf"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}"
    os.makedirs(folder_name, exist_ok=True)
    quoted_folder_name = f'"{folder_name}"'  # Enclose the folder name in double quotes
    command = rf"python -m spacy init config --lang fr --pipeline textcat --optimize efficiency --force {quoted_folder_name}\config.cfg"
    subprocess.run(command, shell=True, check=True)
    print(f"Initialized spaCy configuration for {name} in {folder_name}")

def train_text_classification_model(config_path, output_dir, train_data, val_data):
    # Train the text classification model
    train_cmd = [
        "python",
        "-m",
        "spacy",
        "train",
        config_path,
        "--paths.train",
        train_data,
        "--paths.dev",
        val_data,
        "--output",
        output_dir
    ]
    subprocess.run(train_cmd, shell=True, check=True)

def evaluate_text_classification_model(model_path, test_data):
    # Evaluate the text classification model
    evaluate_cmd = [
        "python",
        "-m",
        "spacy",
        "evaluate",
        model_path,
        test_data
    ]
    subprocess.run(evaluate_cmd, shell=True, check=True)

def collect_model_performance(model, data, output_folder):
    # Calculate and collect model performance metrics
    performance = model.evaluate(data)

    # Create and write performance metrics to a text file
    with open(os.path.join(output_folder, 'performance.txt'), 'w') as file:
        file.write(f"Precision: {performance['textcat_a']}%\n")
        file.write(f"Recall: {performance['textcat_r']}%\n")
        file.write(f"F1 Score: {performance['textcat_f']}%\n")

    print(f"Performance metrics saved to {output_folder}/performance.txt")

######### code #########
    
config_names = ['OUV', 'INT', 'CAD', 'NOUV', 'NCAD', 'AG', 'AI', 'TOUS', 'AG OUV', 'AG INT', 'AG CAD', 'AI OUV', 'AI INT', 'AI CAD', 'NOUV AG', 'NCAD AG', 'NOUV AI', 'NCAD AI', 'ATOT',\
                'ATOT OUV', 'ATOT INT', 'ATOT CAD', 'DATE']


# Train and evaluate models
for name in config_names:
    print("=====================================")
    print(f"Training model for {name}")
    print("=====================================")

    # Load the blank French model
    nlp = spacy.blank("fr")  # Create a blank Language class
    
    #create the storing folder and initialize the config file
    folder = create_folder_and_init(name)
    
    # Import data
    data = import_label_studio_data(r"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\data\training_json\data449.json")

    # Treat data
    data = spacy_to_dataframe(data)
    data = dummy_label(data, name)
    data = clean_dataset(data)
    data = create_tuples(data)

    # Split data
    train_data, val_data, test_data = split_data(data)  # You need to define 'dataset'

    # Skip training if the length of train_data is less than 100
    if len(train_data) < 75:
        print(f"Skipping training for {name} as the training data size is less than 75.")
        print("The length of the training dataset is:", len(train_data))
        continue

    # Convert data to spaCy's binary format
    convert(train_data, fr"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\train.spacy""", name)
    convert(val_data, fr"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\val.spacy""", name)
    convert(test_data, fr"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\test.spacy""", name)

    # Train the model
    config_path = rf"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\config.cfg"""
    train_data = rf"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\train.spacy"""
    val_data = rf"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\val.spacy"""
    output_dir = rf"""C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}"""
    train_text_classification_model(config_path, output_dir, train_data, val_data)

    # # Evaluate the model
    # model_path = rf"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\model-best"
    # test_data = rf"C:\Users\garsonj\Desktop\spacy_finetuning\spacy_files\model\classifyer\{name}\test.spacy"
    # collect_model_performance(model_path, test_data, output_dir)

print("=====================================")
print("Done!")
print("=====================================")