import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
import re
class Train():
    def __init__(self):
        pass

    def get_data(self):
        self.data = pd.read_csv("data.csv")
        i=0
        self.list_of_sent=[]
        for sent in self.data['comment'].values:
            filtered_sentence=[]
            sent=self.cleanhtml(sent)
            for w in sent.split():
                for cleaned_words in self.cleanpunc(w).split():
                    if(cleaned_words.isalpha()):
                        filtered_sentence.append(cleaned_words.lower())
                    else:
                        continue
            self.list_of_sent.append(" ".join(filtered_sentence))
        joblib.dump(self.list_of_sent,"cleaned_data.pkl")

    def cleanhtml(self,sentence): #function to clean the word of any html-tags
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', sentence)
        return cleantext

    def cleanpunc(self,sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        return  cleaned

    def train_data(self):
        try:
            tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
            final_counts = tf_idf_vect.fit_transform(self.list_of_sent)
            X_1, X_test, y_1, y_test = train_test_split(final_counts,self.data["label"],test_size=0.3,random_state=42)
            X_tr, X_cv, y_tr, y_cv = train_test_split(X_1,y_1,test_size=0.3,random_state=42)

            clf = RandomForestClassifier(n_estimators=800)
            clf = clf.fit(X_tr, y_tr)
            pred = clf.predict(X_cv)
            #accuracy score

            result = accuracy_score(y_cv, pred)*100
            print("%f is the accuracy of cross validation dataset"%(result))
            pred = clf.predict(X_test)
            result = accuracy_score(y_test, pred)*100
            print("%f is the accuracy of test dataset"%(result))
            joblib.dump(clf,"trained_data.pkl")
        except Exception as e:
            print("Exception is ",str(e))

if __name__=="__main__":
    obj = Train()
    obj.get_data()
    obj.train_data()
    print("training finished")
