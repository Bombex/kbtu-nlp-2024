Overview:
You are tasked with building a text classification model that can accurately classify 10-K filings based on the industry sector they pertain to (pharma, banks, oilgas). The 10-K filings are annual reports filed by publicly traded companies to give a comprehensive summary of the company's financial performance. You have a dataset composed of pre-cleaned and parsed 10-K filings in text format.

Objectives:

- Explore the data to understand the distribution of classes and common words used in each class.
- Train a classification model using the provided data.
- Validate your model's performance using accuracy and other relevant metrics.
- Test your model on the provided test data.

Instructions:

1. Data Preprocessing:  
    a. Load the textual data from the appropriate folders (pharma, banks, oilgas) for training and validation sets.  
    b. Perform any additional cleaning if necessary (e.g., removing stop words, stemming, etc.).   
    c. Convert the text data into a suitable numeric form using techniques like TF-IDF or word embeddings.
2. Data Exploration:  
    a. Analyze the distribution of documents in each class.  
    b. Find out the most frequent and unique terms in each class.
3. Training the classifier:  
    a. Choose an appropriate machine learning model for text classification (for example, Naive Bayes, SVM, or a neural network).  
    b. Train your model using the training data set.
4. Validation:  
    a. Evaluate your model performance using the validation data set.  
    b. Calculate and report relevant metrics such as accuracy, precision, recall, F1-score.  
    c. Use the results to refine and improve your model.
5. Testing:  
    a. Once satisfied with model performance, apply the final model to the test data set.  
    b. Report your model's performance in the same metrics used for validation.
6. Documentation:  
    a. Throughout the process, document your steps, observations, and the reasons for each decision made.  
    b. Present a final report summarizing methodologies, results, and any conclusions drawn from the task.
