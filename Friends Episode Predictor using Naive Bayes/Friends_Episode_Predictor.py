import os, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
dataset = {}

for each_file in os.listdir('data'):
    content = ''
    with open (os.path.join('data', each_file)) as file:
        content = file.read()
        content = re.sub('\d+\n.*-->.*', '',  content)
        content = re.sub('\s\s\s+', '\n',  content)  
        content = re.sub('\n', ' ',  content)  
    
    
    with open (os.path.join('formatted_data', each_file), 'w+') as file:
        file.write(content)

    dataset[each_file] = content

target = []
data = []

for key, value in dataset.items():
    target.append(key)
    data.append(value)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
print (X_train_counts.shape)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, target)

docs_new = ["joey bets to eat full turkey on thanks giving"]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%s' % (category))