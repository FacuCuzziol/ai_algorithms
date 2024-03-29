# ai_algorithms
A recopilation of some software made by Facundo Cuzziol and Alejandro Nadal on AI models over time.

---

# Perceptron:
Handcrafted perceptron for a specific dataset. This small example has yet to be generalize to work with larger datasets

Made with:

[![L](https://skillicons.dev/icons?i=rust)](https://skillicons.dev)

Code by Alejandro Nadal

---

# Naive Bayes
Simple NBClassifier module that runs predictions on a given dataset, based on the Naive Bayes classifier.
It also includes helping functions inside the following modules:
- **Preprocessing**: Includes helper functions for dataset handling tasks, such as class separation and split into train and test sets, 
- **Metrics**: Provides helper functions for model evaluation

Made with:

[![L](https://skillicons.dev/icons?i=python)](https://skillicons.dev)

## Streamlit app
The Classifier can be accessed at [this streamlit app](https://ai-algorithms-naive-bayes.streamlit.app/) 
## How to run
The main.py file can be executed with streamlit to display the steps that can be followed to run the Classifier. To run this, first install the required libraries with the following command:
```
pip install -r requirements.txt
```

(OPTIONAL): Create a virtual enviroment to install these libraries, so as to avoid conflicts with other versions you might already have installed.

```
streamlit run main.py
```

Made in Python

Code by Facundo Cuzziol

### Changelog
**1.1 - 2023-06-18**
#### Added
- Streamlit sample app to preview steps for Classifier

**1.0 - 2021-12-12**
#### Added
- NBClassifier, preprocessing and metrics module

---
