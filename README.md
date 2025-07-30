# Twitter-Sentiment-Analysis-Using-Neural-Networks


**Dataset Link:**
https://www.kaggle.com/datasets/ferno2/training1600000processednoemoticoncsv


This project focuses on performing sentiment analysis on tweets using a deep learning model built with Keras and TensorFlow. The main objective is to classify tweets as positive or negative, leveraging text preprocessing techniques and a neural network classifier.
📌 Key Features:

    Dataset: A CSV file containing tweets with labeled sentiments (0 for negative, 1 for positive).

    Text Preprocessing:

        Lowercasing, removing URLs, punctuation, and stopwords.

        Tokenization and padding of tweet sequences for model input.

    Model Architecture:

        Embedding layer to represent words in dense vector space.

        LSTM (Long Short-Term Memory) layer for learning temporal dependencies.

        Dense layers with sigmoid activation for binary classification.

    Training & Evaluation:

        Model trained on processed tweets with validation split.

        Accuracy and loss metrics visualized over epochs.

🛠️ Tools & Libraries:

    Python (Jupyter Notebook)

    Keras & TensorFlow

    NLTK & re (for text processing)

    Matplotlib (for visualization)

📊 Output:

    Training accuracy and loss plotted.

    Final evaluation on test data.
