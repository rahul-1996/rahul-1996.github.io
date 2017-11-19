---
title: "Building a Fake News Classifier"
layout: post
date: 2017-11-19 18:30
headerImage: false
tag:
- Fake News Classifier
- Machine Learning
- Python 
- Neural Networks
category: blog
author: rahulmayuranath
description:  Building a fake news classifier in python
---
## Introduction to the problem

The advent of social media has enabled news to spread to a global audience in a matter of seconds, bypassing the usual verification procedures employed by news outlets in the past. As such, the past year had seen a phenomenal increase in spurious articles and features making their way to the social media space. This, coupled with the intentional attempts by spammers and foreign powers to spread misinformation among the voting populace was highlighted during the US Presidential Elections of 2016. Voters across the country were duped by fabricated stories and satirical news misinterpreted as factual, compromising the electoral process.
	
Given the size of the problem at hand, an automated classification system would undermine the effects of spurious or irrelevant news in media feeds. This system tries to help users make more informed decisions on which news stories to believe and which should be taken with a grain of salt. To do so, our system will be trained on a dataset consisting of headlines and articles which have a mix of true and fake headlines. Some of these fake headlines and articles have been derived from satirical sources, while the rest from websites known to push false information. With the intention of making this system quick and efficient, the model is also trained with the headlines only, which is always easier to scrape. Also, in a real-world use case application, the headline could conceivably be derived from the URL of the website itself. We also looked at the variation in the accuracy of our models when using just the headlines versus just the body of the article.

Our ultimate objective for this project is to effectively and automatically label news articles as fake or real. The models assign labels to news articles based on the textual contents of the headlines and text bodies separately. The dataset for the problem consists of around 30,000 entries, conforming to a format of Article title, Text body and associated label. The dataset was assembled by sampling recent fake and news articles datasets from across the internet.
	
We assumed that accurate results could be inferred from just textual analysis. Although some previous work suggests that semantic disambiguation is just as important, we believed that fictitious or ‘fake’ articles would have a clear repetitive use of specific vocabulary. It was reasonably expected that spurious news generated in the backlash of the US Presidential Election would have higher references towards either of the final presidential candidates, and their alleged controversies.

### APPROACH

![alt text](https://i.imgur.com/iRyVIam.png "Approach used")

Corporations like Facebook and Twitter have faced severe backlash for being unable to filter the content that streams on their respective platforms. The community took up the challenge to build reliable models to combat this issue over the past year. This inspired us to try our hand at solving this problem. 
We built our dataset by sampling from multiple datasets online.
This was necessary since a single data source lacked the variety that we needed. Our dataset turned out to be skewed since we had more fake articles. To fix this, we found other datasets of authentic news in a number of categories and built a final dataset containing 30,000 articles.

### Dimensionality Reduction: 
We eyeballed the data and decided we only needed the headline and the news article. We removed features such as source URL because this would overly simplify the task at hand. Removed fields such as likes, comments and shares because these are not indicative of authentic news. Satirical and fictitious articles have been trending on social media and as such these metrics cannot be used to distinguish between real and fake.
For initial analysis, we built a word cloud of the fake and real corpora. The word cloud debunked our initial analysis that spurious articles would have more references to the US Presidential Election candidates but these references appeared equally in both the word clouds.
Next, we vectorized our dataset using unigram TF-IDF. This is a weighted measure of how often a particular phrase occurs in a document relative to how often the phrase occurs across all documents in a corpus. This helps us to focus on the words in the articles and headlines that convey meaning. We use SKLearn to calculate the TFIDF for each word within each document and build a sparse matrix of the resulting features. We did not experiment with different methods or thresholds for selecting the terms included in the vocabulary, or with different lengths of n-grams, but this may be an area to explore in future work.​
Next logical step is to split our data into test and training sets. We employed k-fold cross validation for the same. In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The advantage of this method over repeated random sub-sampling (see below) is that all observations are used for both training and validation, and each observation is used for validation exactly once. [ Quoted from Wikipedia ]
Next, a number of classifier models were built on this dataset, namely SVM, Logistic Regression, Random Forest, Naive Bayes Classifier and an LSTM Neural Network. 
Results were interpreted and our models were pickled so that we don\'t have to train it again each time.

## EXPERIMENTATION AND RESULTS

![alt text](https://i.imgur.com/18acSIH.png "Fake News word cloud")

![alt text](https://i.imgur.com/pFFCq02.png "Real News word cloud")

	

Our preliminary investigation involved building separate word clouds for articles labelled as ‘real’’ and ‘fake’. The ‘nltk’ library in python has a corpus of stop words in English. This was used to clean our data by removing the stop words before the clouds were generated.
The purpose of these word clouds was to test our assumption that spurious articles generated in the wake of the US Presidential elections(2016) would have more references to the candidates or controversy they were embroiled in. The word clouds on the articles could not validate our assumption since these words occurred equally in both authentic and fictitious articles. What we did notice however, is that in the word clouds of headlines of fake news, all the words were related to politics, and in the real news it had references to other topics, such as Entertainment (Christian Bale is referenced).
	
The next step involved splitting the complete dataset of around 30,000 articles  into a train-test split of 80:20, and performing TF-IDF separately on the news headlines and news bodies to generate two separate feature vectors. Stop words were removed while performing TF-IDF. The weights for each feature were scaled logarithmically, and only those features whose frequencies fell within the specified range were chosen.
	
The feature vectors obtained from TF-IDF were used to build the Naive Bayes, SVM, Logistic Regression, Random Forests and LSTM models.
	
The Naive Bayes classifier attempts to label articles into either category by utilizing a Bayesian probabilistic model that assumes independence among the features. The conditional probability of the document belonging to either label is calculated and a label is assigned.
Naive Bayes model was chosen as a baseline for reference, owing to its simplicity. For the title, we got an accuracy of 68.52\%, while the body resulted in a 90.23\% accuracy score. Already, the difference in using the body of the article is evident. It also suggests that titles of articles are made to be misleading, and that there are no obvious patterns that differentiate fake or real.

![alt text](https://i.imgur.com/PPqhIWq.png "Naive Bayes")

Support Vector Machines attempt to fit a hyperplane, or line as such, through an n-dimensional vector space to maximally separate points into regions. Each region has a label associated with it. The SVM generated very high accuracies (with high F1 scores, indicating the model was good). The accuracy obtained by using the body was 95.87\% compared to that by using the title is  73.4\%.

![alt text](https://i.imgur.com/uDLP1GD.png "SVM")

Logistic Regression attempts to predict a dependant dichotomous variable from a set of independent variables. This works for our model as we are trying to apply a label, ‘fake’ or ‘real’, to our news articles given that the label depends on the content of news headline and text body. It resulted in an accuracy of 94.6\% by using the body.

![alt text](https://i.imgur.com/OAk2qyv.png "Logistic Regression")


Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees\' habit of overfitting to their training set[ Quoted from wikipedia ]. With an accuracy of 96.42\% by using the headlines and 74.01\%, Random Forests was also a well performing classifier and gave the highest accuracy for the text body. 

![alt text](https://i.imgur.com/8Shs4Wh.png"Random Forests")

	
Long short-term memory (LSTM) block or network is a simple recurrent neural network which can be used as a building component or block (of hidden layers) for an eventually bigger recurrent neural network.The expression long short-term refers to the fact that LSTM is a model for the short-term memory which can last for a long period of time. There are different types of LSTMs, which differ among them in the components or connections that they have.[ Quoted from Wikipedia ] Our results from the Recurrent Neural Network were quite surprising. It was unable to outperform the Random Forests model when classifying documents by text body giving an accuracy of 93.95\%. However it performed miles ahead of the other models when classifying documents by looking at just the news headlines, with an accuracy of 91.37\%. 
	
Traditional machine learning models have a chance of overfitting their results to the training data. In order to avoid this, we performed a K-folds cross validation for each of these models on each feature vector. The K-Folds Cross validation procedure divides the dataset into k ‘folds’ or chunks. It iteratively selects one of these k chunks of the dataset and uses the other (k-1) folds as the training set. This implies that every ‘fold’ or ‘chunk’ in the dataset will appear in the training data (k-1) times with each chunk being used as the testing data. For better visualization, our cross validation procedure was tested by incrementally increasing the size of the dataset, and iteratively applying the k-folds procedure. This was helpful in understanding how the metrics of our models changed as the number of data points increased. 
	
One of the goals of our experiment was to determine whether spurious articles could be recognized from their headlines alone. It was observed that all our traditional machine learning models couldn’t accurately separate the fake and real articles as well as having low recall. This poor performance can probably be attributed to a lack of  dissimilarity between the feature vectors of fake and real articles. Not to mention, we preprocessed our data beforehand, removing stop words and the like, which reduced the number of features for these models. 

Surprisingly enough our RNN was able to accurately label articles by looking at just the headlines alone. We were unable to determine exactly why our LSTM performed so well on headlines and this appears to be an interesting area for further study.
	
## CONCLUSION

Random Forests achieved the highest accuracy of ~96\% for the text body while the LSTM Neural Network performed the best for the article headlines with an accuracy of ~92\%. The difference in using the body of the article is clear, with the difference in the accuracy of each classifier being at least 15 percentage points when using traditional classification systems.  This leads us to two conclusions. One is that article headlines are made to be more misleading and that traditional classification systems are poor when trying to classify textual data with just one sentence, i.e., a very small set of words. 
    
In the future, the classification system could be expanded to implement a stance detection system, as required by the Fake News Challenge [ 10 ]. Stance Detection here refers to the ability of the system to decide whether the text of the article agrees with the headline, disagrees with the headline, discusses the topic of the headline, but does not take a position if the headline and body are unrelated. 

## LIMITATIONS

There are a few limitations to our analysis that prevents broader generalizability.
1. While TF-IDF performs well, we are possibly overfitting to the data from the news cycle that we chose.  
2. With limited verified sources of fake news, we cannot claim that this approach would generalize to unseen sources. 
3. With heavy reliance on term frequencies, we cannot be confident that this approach would generalize across news cycles.  
			

I collaborated with my team mates Ravi Shreyas Anupindi and Hiranmaya Gundu on the above analysis.  
The github repo of the entire implentation is [here](https://github.com/rahul-1996/fake_news_classifier). Thanks for reading!




