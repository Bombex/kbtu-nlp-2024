import os
import random
import re
from glob import glob

import en_core_web_sm
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tqdm import tqdm


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

    return (preprocessed_texts, labels) if not test else preprocessed_texts


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
    lemmatizer = en_core_web_sm.load()
    stop = set(stopwords.words("english"))

    # Step 1: Preprocessing data
    print("Step 1: Preprocessing data")
    X_train, y_train = preprocess_text(
        path_to_folder="/home/akeresh/Desktop/kbtu/kbtu-nlp-2024/HW1/data/train",
        stopwords=stop,
        lemmatizer=lemmatizer,
        shuffle=True,
    )
    X_val, y_val = preprocess_text(
        path_to_folder="/home/akeresh/Desktop/kbtu/kbtu-nlp-2024/HW1/data/validat",
        stopwords=stop,
        lemmatizer=lemmatizer,
    )
    X_test = preprocess_text(
        path_to_folder="/home/akeresh/Desktop/kbtu/kbtu-nlp-2024/HW1/data/test",
        stopwords=stop,
        lemmatizer=lemmatizer,
        test=True,
    )

    # Step 1.1: Create vectors
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), min_df=0.01, max_df=0.8, max_features=10000
    )
    X_train_features = vectorizer.fit_transform(X_train).todense()
    X_val_features = vectorizer.transform(X_val).todense()
    X_test_features = vectorizer.transform(X_test).todense()

    # Step 2: Training a linear classifier
    print("Step 2: Training a linear classifier")
    model = SVC()
    model.fit(np.asarray(X_train_features), y_train)

    # Step 3: Validate the model performance
    print("Step 3: Validate the model performance")
    validate_classifier(model, X_val_features, y_val)

    # Optional: Tune the model if validation results are not satisfactory

    # Step 4: Test the model
    print("Step 4: Test the model")
    predictions = test_classifier(model, X_test_features)

    # Step 5: Save the predictions
    print("Step 5: Save the predictions")
    save_predictions(
        predictions, "/home/akeresh/Desktop/kbtu/kbtu-nlp-2024/HW1/data/predictions.txt"
    )

    # Additional: Write a report documenting your process and findings
