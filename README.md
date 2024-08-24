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
## File Description
### Step1ofbert.ipy:
#### Data Loading & Preprocessing:
Loaded the dataset from a JSON file.
Combined reviewText and summary into a context column.
Took a 10% sample of the data for analysis.
Used TfidfVectorizer to convert the context into a TF-IDF matrix.
#### Output:
Saved sampled data and TF-IDF vectors to CSV files.
Saved the trained TF-IDF vectorizer as a .pkl file for future use.
### Step2ofbert.ipy:
#### Clustering:
Loaded the TF-IDF vectorized data.
Applied KMeans clustering to categorize the reviews into different teams.
#### Output:
Mapped clusters to specific team names.
Created a final dataset with the review text and corresponding team names.
Saved the final dataset to a CSV file.
### Step3ofbert.ipy:
#### Model Training:
Loaded the final dataset and created mappings for label names.
Tokenized the text data and prepared it for BERT model training.
Set up a BERT model for sequence classification.
Trained the model using the Trainer API from Hugging Face's Transformers library.
#### Model Deployment:
Created a prediction function to classify new text data.
Saved the trained model and tokenizer for later use.
### page.py:
Streamlit Application:
Set up a web application using Streamlit.
Implemented two main pages: Customer Service (for complaint submission and status checking) and Admin (for viewing complaints and team statuses).
Integrated the BERT model to automatically assign teams to incoming complaints.
Used a Lottie animation for visual enhancement.
### Animation - 1723210002909.json:
#### Lottie Animation:
A JSON file containing a Lottie animation used in the Streamlit app for visual appeal.
