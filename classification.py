################################### initialize ####################################################

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import copy
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

colnames=['Text', 'Label']
df_gen = pd.read_csv('C:\\Users\\abhin\\AUDProject\\general.csv', usecols=colnames)     # for general data
df_car = pd.read_csv('C:\\Users\\abhin\\AUDProject\\cars.csv', usecols=colnames)     # for car data
df_gen.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df_car.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
reviews_gen = df_gen['Text'].tolist()
reviews_car = df_car['Text'].tolist()
tokens, lemmatized = [], []

lemmatizer = nltk.stem.WordNetLemmatizer()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
vectorizer = CountVectorizer(min_df=5, ngram_range=(1,2), stop_words='english')

###################################################################################################

######################################## Preprocess ###############################################

reviews_gen = [x for x in reviews_gen if not isinstance(x, float)]
reviews_car = [x for x in reviews_car if not isinstance(x, float)]
def pre_process_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    lemmatized = " ".join(lemmatized)
    return lemmatized

reviews_gen = list(map(pre_process_text,reviews_gen))
reviews_car = list(map(pre_process_text,reviews_car))

n1 = df_gen.columns[0]
df_gen.drop(n1, axis=1, inplace=True)
df_gen[n1] = reviews_gen
n2 = df_car.columns[0]
df_car.drop(n2, axis=1, inplace=True)
df_car[n2] = reviews_car

data_gen, data_car = df_gen.values.tolist(), df_car.values.tolist()
train_gen, test_gen = data_gen[:1500] , data_gen[1500:]  # split train and test set for restaurants
train_car, test_car = data_car[:1500], data_car[1500:]  # split train and test set for movies
training, testing = train_gen + train_car, test_gen + test_car  # combine training and test sets
x_train, x_test = [i[1] for i in training], [i[1] for i in testing]
y_train, y_test = [i[0] for i in training], [i[0] for i in testing]
# dummy function for pre processing
def tk(doc):
    return doc

tfidf = TfidfVectorizer(analyzer='word', preprocessor=tk, tokenizer=tk, ngram_range=(1, 2),
                        min_df=5, stop_words='english') # initialize TF-IDF vectorizer
tfidf.fit(x_train)
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)

###################################################################################################

###########################################  Modelling ############################################

# Logit
Logitmodel = LogisticRegression()
Logitmodel.fit(x_train, y_train)
y_pred_logit = Logitmodel.predict(x_test)
acc_logit = accuracy_score(y_test, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))

# Naive Bayes
NBmodel = MultinomialNB()
NBmodel.fit(x_train, y_train)
y_pred_NB = NBmodel.predict(x_test)
acc_NB = accuracy_score(y_test, y_pred_NB)
print("Naive Bayes model Accuracy::{:.2f}%".format(acc_NB*100))

# Support Vector Classifier
SVMmodel = LinearSVC()
SVMmodel.fit(x_train, y_train)
y_pred_SVM = SVMmodel.predict(x_test)
acc_SVM = accuracy_score(y_test, y_pred_SVM)
print("SVM model Accuracy:{:.2f}%".format(acc_SVM*100))

# Random Forest Classifier
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy',
                                 bootstrap=True, random_state=0) ## number of trees and number of layers/depth
RFmodel.fit(x_train, y_train)
y_pred_RF = RFmodel.predict(x_test)
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))

# Deep Learning
DLmodel2 = MLPClassifier(solver='adam', hidden_layer_sizes=(60,50,40), random_state=1, n_iter_no_change=20,
                        learning_rate='adaptive', max_iter=200, verbose=False, early_stopping=True)
DLmodel2.fit(x_train, y_train)
y_pred_DL= DLmodel2.predict(x_test)
acc_DL = accuracy_score(y_test, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))

###################################################################################################

############################################ LSTM #################################################

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
MAX_NB_WORDS = 50000 # The maximum number of words to be used. (most frequent)
MAX_SEQUENCE_LENGTH = 250 # Max number of words in each complaint.
EMBEDDING_DIM = 100 # This is fixed.

df = pd.concat([df_gen, df_car])
df['Text'] = df['Text'].astype(str)
df['Text'] = df['Text'].apply(clean_text)
df['Text'] = df['Text'].str.replace('\d+', '')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['Label']).values
word_index = tokenizer.word_index
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

print('Found %s unique tokens.' % len(word_index))
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(140, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 60
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

###########################################################################################

# Convolution
# model.add(Dropout(0.22))
# model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
# model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
# model.add(Dense(2))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# kernel_size = 5
# filters = 64
# pool_size = 4

################################### TEST #################################################

tippe_gen = pd.read_csv('C:\\Users\\abhin\\AUDProject\\tippe_gen.csv', usecols=colnames)     # for general data
tippe_car = pd.read_csv('C:\\Users\\abhin\\AUDProject\\tippe_car.csv', usecols=colnames)
# indy_gen = pd.read_csv('C:\\Users\\abhin\\AUDProject\\indy_gen.csv', usecols=colnames)     # for general data
indy_car = pd.read_csv('C:\\Users\\abhin\\AUDProject\\indy_car.csv', usecols=colnames)
final_df = pd.concat([tippe_car, indy_car])
final_df['Text'] = final_df['Text'].astype(str)
final_df['Text'] = final_df['Text'].apply(clean_text)
final_df['Text'] = final_df['Text'].str.replace('\d+', '')
tokenizer.fit_on_texts(final_df['Text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X1 = tokenizer.texts_to_sequences(final_df['Text'].values)
X1 = pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
Y1 = pd.get_dummies(final_df['Label']).values
print('Shape of label tensor:', Y1.shape)
accr = model.evaluate(X1,Y1)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

############################################################################################

def TFIDF(ran,min, train, test):

    def tk(doc):
        return doc
    ran = (1,ran)
    print(ran)
    tfidf = TfidfVectorizer(analyzer='word', preprocessor=tk, tokenizer=tk, ngram_range=ran,
                            min_df=5) # initialize TF-IDF vectorizer
    tfidf.fit(train)
    x_train = tfidf.transform(train)
    x_test = tfidf.transform(test)
    print("For min_df=",min,"and ngram=",ran)
    return x_train,x_test


def deep(i,j,x_train,y_train,x_test,y_test):

    DLmodel2 = MLPClassifier(solver='adam', hidden_layer_sizes=(i,j), random_state=1, n_iter_no_change=20,
                             learning_rate='adaptive', max_iter=200, verbose=False, early_stopping=True)
    DLmodel2.fit(x_train, y_train)
    y_pred_DL = DLmodel2.predict(x_test)
    acc_DL = accuracy_score(y_test, y_pred_DL)

    print("For nodes = ",i,"and layers = ", j," accuracy=  ",acc_DL)
     return acc_DL


for min_df in [3,4,5]:
    for n_gram in [2,3,4]:
        train,test = TFIDF(n_gram,min_df,x_train,x_test)
        for nodes in list(np.arange(5,40,5)):
            for layers in [2,3,4,5,6]:
                acc = deep(nodes,layers,train,y_train,test,y_test)
                if acc>best:
                    best=acc
                    best_min_df=min_df
                    best_ngram = n_gram
                    best_nodes = nodes
                    best_layers = layers

def tk(doc):
    return doc


tfidf = TfidfVectorizer(analyzer='word', preprocessor=tk, tokenizer=tk, ngram_range=(1,4),
                            min_df=3)

tfidf.fit(x_train)
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(df_x)



# Deep Learning
DLmodel2 = MLPClassifier(solver='adam', hidden_layer_sizes=(15,3), random_state=1, n_iter_no_change=20,
                        learning_rate='adaptive', max_iter=200, verbose=False, early_stopping=True)
DLmodel2.fit(x_train, y_train)
y_pred_DL= DLmodel2.predict(x_test)
acc_DL = accuracy_score(df_y, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))
