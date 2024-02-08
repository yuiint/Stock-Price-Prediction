import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.metrics import accuracy_score  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix    
from plotly.subplots import make_subplots   
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


import logging
logging.basicConfig(format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info('started')


def get_X_y(df, term_type):
    documents = df['cut_title_content_term'].to_list()
    vectorizer = TfidfVectorizer(vocabulary=term_type, use_idf=True)

    # Fit and transform the documents
    X = vectorizer.fit_transform(documents)
    y = df['type']

    return vectorizer, X, y


def predict(X, y, model_name: str, classifier):
    '''
    model; model name
    classifier: classifier for the model
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_test.toarray())

    print(f'Model: {model_name}\n')
    acc_score = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {round(acc_score, 4)}")
    print('='*50)
    print(classification_report(y_test, y_pred, zero_division=0))
    print('='*50)
    cm = confusion_matrix(y_test, y_pred, labels=['up','down'])
    print(cm)
    print('='*50)

    #  Cross Validation
    if model_name != 'KNN': #  KNN 現在用不出來 QQ
        print('Cross Validation')
        scores = cross_val_score(classifier,X.toarray(),y,cv=5,scoring='accuracy') #交叉驗證，計算準確率
        print(scores)
        print(f"Avg. Accuracy: {round(scores.mean(), 4)}")
        print('='*50)
    # Plot Confusion Matrix
    # Create a 1x1 subplot
    fig = make_subplots(rows=1, cols=1)

    # Add heatmap trace to the subplot
    fig.add_trace(
        go.Heatmap(
            x=['Predicted UP', 'Predicted DOWN'],  # x-axis labels
            y=['Actual UP', 'Actual DOWN'],  # y-axis labels
            z=cm,  # Confusion matrix data
            # colorscale='blue',  # Color scale
            reversescale=False,  # Whether to reverse the color scale
            showscale=True,  # Whether to show the color scale
            xgap=2,  # x-axis gap
            ygap=2  # y-axis gap
        )
    )

    # Update the x-axis and y-axis label positions
    fig.update_xaxes(side='bottom')
    fig.update_yaxes(side='left')

    # Set the title and size of the plot
    fig.update_layout(
        title='Confusion Matrix',
        width=500,  # Plot width
        height=400  # Plot height
    )

    # Show the plot
    fig.show()
    return acc_score, y_test, y_pred


def prediction_result(X, y):

    rf_classifier = RandomForestClassifier()
    dt_classifier = DecisionTreeClassifier(criterion="entropy")
    nb_classifier = MultinomialNB()
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    svm_classifier = SVC(kernel='linear')


    scenarios = [
    ('RF', rf_classifier),
    ('DT', dt_classifier),
    ('NB', nb_classifier),
    ('KNN',knn_classifier),
    ('SVM', svm_classifier),
    ]

    for model, classfier in scenarios:
        acc_score, y_test, y_pred = predict(X, y, model, classfier)
 
        if 'vote' not in locals():
            vote = pd.DataFrame({}, 
                index=y_test.index)
        if (acc_score) > 0.85:
            vote[model] = y_pred

    vote = vote.replace({'up':1, 'down':0})
    vote['voting'] = vote.T.sum().apply(lambda x:1 if x >= (vote.shape[1]/2) else 0)

    vote['y_test'] = y_test
    vote['y_test'] = vote['y_test'].replace({'up':1, 'down':0})

    vote_accuracy = (vote['voting'] == vote['y_test']).mean()
    print(f"Vote accuracy = {round(vote_accuracy, 4)}")


  