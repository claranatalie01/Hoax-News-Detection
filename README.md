# Hoax News Detection System - TSDN 2023

The project undertaken by Team Bayez for the Turnamen Sains Data Nasional 2023 confronts a pressing issue in the digital era: the proliferation of hoax news, particularly within the political realm. Below is the updated README including the rationale behind our project and a description of the dataset used.

## Table of Contents

- [Team Members](#team-members)
- [Project Rationale](#project-rationale)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
  - [Machine Learning Process Flow](#machine-learning-process-flow)
  - [Deep Neural Network Process Flow](#deep-neural-network-process-flow)
- [Web Apps Integration](#web-apps-integration)
- [Conclusion and Recommendations](#conclusion-and-recommendations)
- [References](#references)

## Team Members
- Muhammad Iqbal
- Lutfi Lingga Saputra
- Clara Natalie Surjadjajadi
- Sekar Ayu Larasati

## Project Rationale

In the contemporary information landscape, social media and other news channels have been exploited for political gain by certain individuals. These platforms, which should serve as conduits for political literacy, have instead been utilized as vehicles for propaganda and provocation (Amilin, 2019).

According to the Indonesian Ministry of Communication and Information Technology (Kominfo, 2023), between August 2018 and May 2023, there were 11,642 hoax contents identified, with 1,373 of them categorized as political hoaxes.

As per Amilin (2019), mitigating political hoaxes requires appropriate, correct, and precise methods. One such method is for the state to provide intelligent solutions to address the challenges posed by the evolving landscape of information and communication technology. Hoax content often features highly provocative headlines that attract reader attention, leading to provocation and manipulation, such as the use of hoax creation for incitement or phishing through clickable links.

This context underscores the necessity of our project: the development of a Hoax News Detection System to filter out false information and uphold the integrity of news dissemination, particularly in the political sphere.

# Dataset Description

## Valid News Dataset

- **Politics News**:
  - 700 articles from CNN Indonesia
  - 700 articles from Tempo
  - 700 articles from Kompas
  - *Total*: 2,100 articles

- **Mixed News**:
  - Approximately 8,000 articles from various trusted news sources

## Hoax News Dataset
- **Politics Hoax**:
  - Around 2,000 articles categorized under politics from turnbackhoax.id

- **Mixed Hoax**:
  - Approximately 8,000 articles with mixed content from turnbackhoax.id


## Technologies Used

- **Python**: For scripting and creating machine learning models.
- **Sastrawi**: A stemming tool for the Indonesian language.
- **Scikit-learn**: For implementing machine learning algorithms and data preprocessing.
- **Keras with TensorFlow backend**: For building deep learning models.
- **FastText Word Embeddings**: For generating vector representations of words.
- **Pickle**: For serializing and saving the machine learning models.
- **Spyder**: An IDE for Python used for integrating the model into the web application.

## Workflow

### Machine Learning Process Flow

1. **Raw Data**: A dataset consisting of 22,000 entries of political and non-political news from various sources within Indonesia, labeled as hoax or valid.
2. **Data Preprocessing**: Involves removing duplicates, irrelevant words, and stopwords; labeling the dataset; and stemming with Sastrawi.
3. **TF-IDF Vectorizer**: To determine the importance of words within the corpus for classification purposes.
4. **Data Split**: The dataset is split into 80% training and 20% testing, with cross-validation applied.
5. **Model Training and Evaluation**: Machine learning algorithms like Multinomial NB, Passive Aggressive, and Multinomial NB with hyperparameter tuning are evaluated using metrics such as Accuracy, F1, Recall, and Precision.

### Deep Neural Network Process Flow

1. **Tokenization**: Converting words to numeric tokens.
2. **Padding**: Standardizing input sequence lengths.
3. **Fast-text Word Embedding**: Creating an embedding matrix using FastText vectors.
4. **DNN Architecture Model**: Designing a sequential model with various RNN layers.
5. **Training and Evaluation**: Using RNN layers and evaluating models based on performance metrics.
6. **Model Selection**: The LSTM-based model exhibited the best performance for integration into WebApps.

## Web Apps Integration

1. **Model Import**: Import the best-performing model for use in the web application.
2. **Saving the Model**: The model and vectorizer are serialized into `.pkl` or `.h5` formats.
3. **Web Application Development**: Utilize Spyder to build the web app, integrating the AI model with a server-side script, and preparing `.html` and `.css` files for the UI.
4. **Deployment**: The web application can be hosted publicly, allowing users to verify news articles through a browser.

## Conclusion and Recommendations

Based on international scientific journals, DNNs are recommended for hoax detection over traditional ML processes. Specifically, the LSTM model outperforms others in accuracy and reliability. WebApps provide a user-friendly interface for the general public to verify news authenticity.

Future research can expand datasets to include English-language content, source data from social media platforms, and explore other machine learning models like SVM or 1D-CNN for improved performance.

## References

The following works were referenced in the development and analysis conducted within this project:

### Journal Articles

- Amilin, A. (2019). Pengaruh Hoaks Politik dalam Era Post-Truth terhadap Ketahanan Nasional dan Dampaknya pada Kelangsungan Pembangunan Nasional. Jurnal Lemhannas RI, 7(3), 5-11.
- Nayoga, B. P., Adipradana, R., Suryadi, R., & Soehartono, D. (2021). Hoax Analyzer for Indonesian News Using Deep Learning Models. Procedia Computer Science, 179, 704-711.
- Siddiq, N. (2017). Penegakan Hukum Pidana Dalam Penanggulangan Berita Palsu (Hoax) Menurut Undang-Undang No.11 Tahun 2008 Yang Telah Dirubah Menjadi Undang-Undang No.19 Tahun 2016 Tentang Informasi Dan Transaksi. Lex Et Societatis, Vol. V, No. 10, 26-32.

### Websites

- Himslaw Article, "Bahaya Menyebarkan Berita Hoaks." Accessed February 2, 2020. [https://www.himslaw.com/bahaya-menyebarkan-berita-hoaks](https://www.himslaw.com/bahaya-menyebarkan-berita-hoaks)
- Reporters Without Borders (RSF). [https://rsf.org/en/index](https://rsf.org/en/index)
- Disinformation Index. "Disinformation Risk Assessment: The Online News Market in Indonesia." November 2, 2022. [https://www.disinformationindex.org/country-studies/2022-11-02-disinformation-risk-assessment-the-online-news-market-in-indonesia/](https://www.disinformationindex.org/country-studies/2022-11-02-disinformation-risk-assessment-the-online-news-market-in-indonesia/)
- TurnBackHoax.ID. [https://turnbackhoax.id/](https://turnbackhoax.id/)

Each of these sources contributed to the understanding and conceptual framework of our project, providing insights into the prevalence of hoaxes, their detection, and the legal as well as societal responses to such challenges in Indonesia.
