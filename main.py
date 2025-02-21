# This is main file, run this file.

import pandas as pd
from data_loader import DataLoader
from preprocessing import TextPreprocessor
from eda import EDA
from vectorization import Vectorizer
from modeling import Model

def main():
# Load Data
    loader = DataLoader()
    documents = loader.load_documents('data/your_financial_articles_directory/')
    
# Preprocessing
    preprocessor = TextPreprocessor()
    processed_documents = preprocessor.preprocess_documents(documents)
    
# Creating a DataFrame for unlabeled data
    data = pd.DataFrame(processed_documents, columns=['filename', 'tokens'])
    data['text'] = data['tokens'].apply(lambda x: ' '.join(x))  # Join tokens back to text

# Load labeled categories
    categories_df = pd.read_csv('data/categories.csv')

# Vectorization
    vectorizer = Vectorizer()
    X_labeled = vectorizer.fit_transform(categories_df['text'])
    y_labeled = categories_df['category']

# Train the model on the labeled data
    model = Model()
    model.train(X_labeled, y_labeled)

# Predict categories for the unlabeled dataset
    X_unlabeled = vectorizer.transform(data['text'])
    predictions = model.model.predict(X_unlabeled)

# Add predictions to the DataFrame
    data['category'] = predictions

# Display predictions to check results
    print(data[['filename', 'category']])

# Check distribution of predicted categories
    print("Predicted category distribution:")
    print(data['category'].value_counts())

# Exploratory Data Analysis
    eda = EDA(data)
    eda.visualize_data_distribution()
    eda.calculate_statistics()

# Evaluate model performance (optional)
    model.evaluate()

if __name__ == "__main__":
    main()
