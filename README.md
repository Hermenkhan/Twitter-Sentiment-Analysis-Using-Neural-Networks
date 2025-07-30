# Twitter-Sentiment-Analysis-Using-Neural-Networks

This project performs binary sentiment classification on tweets using a deep learning model built with Keras and TensorFlow. It aims to classify tweets as positive or negative based on their content, using an LSTM-based neural network.
📂 Project Structure

    1_lft_Twitter_Sentiment_Analysis_Using_NN_😎(1).ipynb – Main Jupyter Notebook for preprocessing, model building, training, and evaluation.

    tweets.csv – Input dataset containing tweets and corresponding sentiment labels.

📊 Dataset Description

The dataset contains tweets labeled with:

    0 → Negative sentiment

    1 → Positive sentiment

Each entry has:

    Text: The tweet content

    Target: Sentiment label (0 or 1)

⚙️ Installation

    Clone this repository

git clone https://github.com/your-username/twitter-sentiment-nn.git
cd twitter-sentiment-nn

Create and activate a virtual environment (optional but recommended)

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

Install the required libraries

pip install -r requirements.txt

If requirements.txt is not provided, install manually:

pip install numpy pandas nltk matplotlib tensorflow scikit-learn

Download NLTK stopwords (first-time use only)
Inside the notebook or Python shell:

    import nltk
    nltk.download('stopwords')

🧹 Preprocessing Steps

    Lowercase text conversion

    Removal of:

        URLs

        Punctuation

        Digits and special characters

    Stopword filtering using NLTK

    Tokenization and sequence padding using Keras' Tokenizer

🧠 Model Architecture

    Embedding Layer: Converts tokens into dense vectors

    LSTM Layer: Captures temporal and semantic relationships

    Dense Layers:

        Fully connected layers

        Final sigmoid activation for binary classification

📈 Training & Evaluation

    Model trained with:

        Binary cross-entropy loss

        Adam optimizer

        20% validation split

    Metrics:

        Accuracy and loss plotted across epochs

        Final evaluation on unseen test data

📌 Sample Output

Training Accuracy: 92%
Validation Accuracy: 89%
Test Accuracy: 90%

Plots of training vs. validation accuracy and loss are generated at the end of the notebook.
🚀 Future Improvements

    Use more advanced architectures like BERT or transformers

    Expand to multi-class sentiment (e.g., neutral)

    Deploy model with a web API or web app (e.g., using Flask or FastAPI)

**Dataset Link:**
https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv
