Natural Language Processing deals with text data.

### Pipeline

```
1. Data Storage (MongoDB)
2. Data Collection (Raw_Data)
3. Data Validation (Schema)
4. Data Storage (Preprocessed)
5. Feature Engineering
6. Model Building
7. Evaluation
8. Model Deployment-flask
9. Orchestration-prefect
```

### Orchestration-prefect

```
-prefect server start
-prefect deployment build main.py:main_flow -n demo-deployment
```

# Topic Modeling (Text Classification) Project using NLP

Worked on an exciting natural language processing (NLP) project that involved text classification using machine learning techniques. In this project, I aimed to classify news articles into different categories.

Here's a breakdown of the key steps and findings of the project:

**1. Data Preparation:**

- I started by loading the dataset from a CSV file, which contained news articles and their corresponding categories.

- I explored the dataset's structure and identified the unique categories in it.

- The dataset comprised various categories such as politics, business, entertainment, and more.

**2. Feature Engineering:**

- To prepare the text data for machine learning, I used the CV (Count-Vectorizer: count frequency) vectorization technique. It converted the text data into numerical features, considering the importance of words in each article.

- I limited the feature set to the top 5,000 terms and removed common English stop words to improve the quality of the features.

**3. Model Building:**

- I divided the dataset into training and testing sets to evaluate the model's performance effectively.

- For text classification, I tried various models such as linearSVC, MultinomialNB and Support Vector Machines with different kernel functions to identify the best-performing model.

- The different models with parameters to show the highest accuracy on the test data.

**4. Model Evaluation:**

- I measured the model's performance using the accuracy score, which indicated the proportion of correctly classified articles.

- The models improve accuracy on the test data, demonstrating its effectiveness in classifying news articles accurately.

**5. Inference and Deployment:**

- I demonstrated how the trained model could be used to classify new headlines into their respective categories. For example, I input headlines related to business person and actor and showed that the model correctly predicted the categories of "business" and "entertainment," respectively.

**6. Real-World Application:**

- I highlighted the practical applications of this project, which extend beyond news categorization. Such techniques can be employed in recommendation systems or information retrieval, among other fields.

In conclusion, this project showcased the power of natural language processing and machine learning in automating the categorization of news articles. The skills and knowledge gained from this project can be applied to a wide range of applications in the field of data science and NLP. Any question please reach out.

### Dockerfile

```- application build: docker build -t hsinghsudwal/nlp_flask .
- container: docker run -d -p 80:80 hsinghsudwal/nlp_flask
- repo: docker push hsinghsudwal/nlp_flask
```

### Mlflow Dagshub

`Experiments`
