from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.externals import joblib
import sys
class Test():
    def __init__(self,parameters):
        self.query = parameters.get("query","")

    def clean_data(self):
        i=0
        self.list_of_sent = joblib.load("cleaned_data.pkl")
        for sent in [self.query]:
            filtered_sentence=[]
            sent=self.cleanhtml(sent)
            for w in sent.split():
                for cleaned_words in self.cleanpunc(w).split():
                    if(cleaned_words.isalpha()):
                        filtered_sentence.append(cleaned_words.lower())
                    else:
                        continue
        self.list_of_sent.append(" ".join(filtered_sentence))

    def cleanhtml(self,sentence): #function to clean the word of any html-tags
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', sentence)
        return cleantext
    def cleanpunc(self,sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        return  cleaned

    def test_data(self):
        try:
            tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
            final_counts = tf_idf_vect.fit_transform(self.list_of_sent)
            clf = joblib.load("trained_data.pkl")
            pred = clf.predict(final_counts[-1])
            print("Question :",self.query)
            print("predicted :",pred[0])
        except Exception as e:
            print("Exception is ",e)

if __name__=="__main__":
    args = sys.argv[1]
    parameters = {"query":args}
    obj = Test(parameters)
    obj.clean_data()
    obj.test_data()


