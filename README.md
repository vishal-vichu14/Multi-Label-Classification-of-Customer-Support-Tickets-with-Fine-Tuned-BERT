# Multi-Label-Classification-of-Customer-Support-Tickets-with-Fine-Tuned-BERT
This project involves the implementation of a multi-label classification model to automatically categorize customer support tickets based on their textual content. The model is built using the BERT (Bidirectional Encoder Representations from Transformers) architecture, fine-tuned on a labeled dataset of customer support tickets.
## Packages to install
1. pip install pandas
2. pip install scikit-learn
3. pip install scipy
4. pip install joblib
5. pip install transformers
6. pip install torch
7. pip install streamlit
8. pip install streamlit-lottie
9. pip install numpy
```python
pip install pandas scikit-learn scipy joblib transformers torch streamlit streamlit-lottie numpy
```
## Dataset Description
https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews
Amazon Reviews Dataset: This is a subset of Amazon reviews from the Electronics category, spanning May 1996 - July 2014.
## File Description
### Step1ofbert.ipy:
#### Data Loading & Preprocessing:
* Load Data: The code reads JSON data from a specified file path into a Pandas DataFrame and prints a confirmation message.
* Combine Columns: It creates a new column 'context' by concatenating 'reviewText' and 'summary' to provide a fuller context for each review.
* Sample Data: It selects a 10% sample of the data (specified size) for further analysis and extracts the 'context' column from the sampled data.
* Preprocess Text: It defines and applies a preprocessing function to clean and prepare the text data. This function converts text to lowercase, removes short words and punctuation, tokenizes, removes stopwords, applies stemming, and performs lemmatization.
* Vectorize Text: It uses a TF-IDF Vectorizer to convert the preprocessed text data into numerical features, which are then stored in a DataFrame.
* Save Results: It saves the sampled data, TF-IDF features, and the vectorizer model to CSV and pickle files for future use.
#### Output:
Saved sampled data and TF-IDF vectors to CSV files.
Saved the trained TF-IDF vectorizer as a .pkl file for future use.
### Step2ofbert.ipy:
#### Clustering:
* Load Data: The code reads the TF-IDF vectorized data and the sampled data (containing original text) from CSV files.
* Convert to Sparse Matrix: It converts the TF-IDF DataFrame into a sparse matrix format, which is suitable for clustering algorithms like KMeans.
* Initialize KMeans: It sets up the KMeans clustering model with a specified number of clusters and fits the model on the sparse matrix of TF-IDF vectors.
* Assign Clusters: It assigns cluster labels to the sampled data based on the KMeans model and counts the number of data points in each cluster.
* Analyze Clusters: It loads the previously saved TF-IDF vectorizer, retrieves feature names, and identifies the top terms for each cluster by examining the cluster centroids.
* Map Teams and Save Results: It maps cluster labels to specific team names, adds these team names to the sampled data, and creates a final DataFrame with review text and assigned team names, which is then saved to a CSV file.
#### Output:
Mapped clusters to specific team names.
Created a final dataset with the review text and corresponding team names.
Saved the final dataset to a CSV file.
### Step3ofbert.ipy:
#### Model Training:
* Load and Prepare Data:The code starts by loading the dataset from a CSV file. It then extracts unique team names to create mappings from label IDs to team names (id2label) and vice versa (label2id). Numeric labels are assigned to each team name in the dataset using pd.factorize.
* Initialize BERT Model:The BERT tokenizer and sequence classification model are loaded using the transformers library. The model is configured with the number of unique labels and the mapping dictionaries for label IDs.
* Data Preparation:The review text is converted to string format and split into training, validation, and test sets. Each set is tokenized using the BERT tokenizer, and DataLoader objects are created for each set using a custom ReviewsDataset class.
* Training Setup:Training arguments for the BERT model are defined, including parameters like the number of epochs, batch sizes, learning rate settings, and logging configurations. A Trainer object is initialized with the model, training arguments, and datasets.
* Model Training and Evaluation:The model is trained using the Trainer.train() method. After training, the model is evaluated on the training, validation, and test datasets. Evaluation metrics like accuracy, precision, recall, and F1 score are computed.
*Prediction and Saving:A prediction function is defined to classify new text data using the trained model. The model and tokenizer are saved for future use, and they are later reloaded from the saved files. A sentiment analysis pipeline is created for making predictions.
### page.py:
#### Streamlit Application:
Set up a web application using Streamlit.
Implemented two main pages: Customer Service (for complaint submission and status checking) and Admin (for viewing complaints and team statuses).
Integrated the BERT model to automatically assign teams to incoming complaints.
Used a Lottie animation for visual enhancement.
### Animation - 1723210002909.json:
#### Lottie Animation:
A JSON file containing a Lottie animation used in the Streamlit app for visual appeal.
