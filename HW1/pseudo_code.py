import os
import random
import re
from glob import glob

import en_core_web_sm
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tqdm import tqdm
import argparse

nltk.download('stopwords')

def clean_text(review, stopwords):
    review = review.lower()  # to lowercase
    review = re.sub("[^a-zA-Z ]+", "", review)  # delete punctuation
    review = " ".join([word for word in review.split() if word not in stopwords])

    return review


def lemmatize_text(text, lemmatizer):
    spacy_results = lemmatizer(text)
    return " ".join([token.lemma_ for token in spacy_results])


def preprocess_text(
    path_to_folder: str,
    stopwords,
    lemmatizer,
    pattern: str = "**/*.txt",
    test: bool = False,
    shuffle: bool = False,
):
    preprocessed_texts, labels = [], []
    if test:
        pattern = "*.txt"
    paths = glob(os.path.join(path_to_folder, pattern))
    if shuffle:
        random.shuffle(paths)
    for path in tqdm(paths):
        with open(path, "r") as file:
            text = file.readline().strip()
            cleaned_text = clean_text(text, stopwords)
            lemmatized_text = lemmatize_text(cleaned_text, lemmatizer)
            preprocessed_texts.append(lemmatized_text)

        if not test:
            label = os.path.basename(os.path.dirname(path))
            labels.append(label)
    
    file_names = [os.path.basename(path) for path in paths]
    if not test:
        df = pd.DataFrame({"file_names": file_names, "preprocessed_texts": preprocessed_texts, "labels": labels})
    else:
        df = pd.DataFrame({"file_names": file_names, "preprocessed_texts": preprocessed_texts})
    # return (preprocessed_texts, labels) if not test else preprocessed_texts
    return df


def validate_classifier(model, X_val, y_val):
    print(classification_report(y_val, model.predict(np.asarray(X_val))))


def test_classifier(model, X_test):
    y_test = model.predict(np.asarray(X_test))
    return y_test


def save_predictions(predictions, output_path):
    with open(output_path, "w") as f:
        for item in predictions:
            f.write("%s\n" % item)


# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for classification text')
    parser.add_argument('--train', type=str, help='Path to train data folder')
    parser.add_argument('--val', type=str, help='Path to validation data folder')
    parser.add_argument('--test', type=str, help='Path to test data folder')
    parser.add_argument('--prediction', type=str, help='Path to save predictions', 
                        required=False, default=os.path.join(os.getcwd(), "predictions.txt"))
    args = parser.parse_args()


    lemmatizer = en_core_web_sm.load()
    stop = set(stopwords.words("english"))

    # Step 1: Preprocessing data
    print("Step 1: Preprocessing data")
    df_train = preprocess_text(
        path_to_folder=args.train,
        stopwords=stop,
        lemmatizer=lemmatizer,
        shuffle=True,
    )
    df_val = preprocess_text(
        path_to_folder=args.val,
        stopwords=stop,
        lemmatizer=lemmatizer,
    )
    df_test = preprocess_text(
        path_to_folder=args.test,
        stopwords=stop,
        lemmatizer=lemmatizer,
        test=True,
    )

    # Step 1.1: Create vectors
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), min_df=0.01, max_df=0.8, max_features=10000
    )
    X_train_features = vectorizer.fit_transform(df_train["preprocessed_texts"]).todense()
    X_val_features = vectorizer.transform(df_val["preprocessed_texts"]).todense()
    X_test_features = vectorizer.transform(df_test["preprocessed_texts"]).todense()

    # Step 2: Training a linear classifier
    print("Step 2: Training a linear classifier")
    model = SVC()
    model.fit(np.asarray(X_train_features), df_train["labels"])

    # Step 3: Validate the model performance
    print("Step 3: Validate the model performance")
    validate_classifier(model, X_val_features, df_val["labels"])

    # Optional: Tune the model if validation results are not satisfactory

    # Step 4: Test the model
    print("Step 4: Test the model")
    predictions = test_classifier(model, X_test_features)
    df_test["predictions"] = predictions
    

    # Step 5: Save the predictions
    print("Step 5: Save the predictions")
    df_test[["file_names", "predictions"]].to_csv(args.prediction, sep='\t', index=False)
    # save_predictions(
    #     predictions, args.prediction
    # )