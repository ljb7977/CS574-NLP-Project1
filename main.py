import math
import os
import re
import csv
from sklearn.metrics import classification_report

stopwords = []
pos_likelihood = None
neg_likelihood = None
pos_prior = 0
neg_prior = 0

def get_stop_word():
    global stopwords
    f_stopwords = open("stop-word-list.csv", encoding='UTF-8')
    stopwords = f_stopwords.readline().split(", ")
    f_stopwords.close()

def tokenizer(line):
    global stopwords
    words = line.lower()
    words = re.sub("(can't)", "cannot", words)
    words = re.sub("(n't)", "not", words)
    words = re.sub("(<.*?>|[^a-zA-Z'])+", ' ', words).split()

    words = [item for item in words if item not in stopwords]
    return words

def classify_nb():
    global pos_likelihood
    global neg_likelihood
    truepos = 0
    trueneg = 0
    falsepos = 0
    falseneg = 0
    y_pred = []
    y_true = []

    '''
    f = open("pos.csv")
    for key, val in csv.reader(f):
        pos_likelihood[key] = float(val)
    f.close()

    f = open("neg.csv")
    for key, val in csv.reader(f):
        neg_likelihood[key] = float(val)
    f.close()
    '''

    for mode in ("neg", "pos"):
        filepath = "aclImdb/test/" + mode + "/"
        for file in sorted(os.listdir(filepath),
                           key=lambda x: (int(re.sub('\D(.*)', '', x)), x)):
            document = open(filepath + file, encoding='UTF-8')

            words = tokenizer(document.readline())

            new_words = []
            for word in words:
                if word in pos_likelihood.keys():
                    new_words.append(word)

            words = new_words

            # words = list(set(words)&set(pos_likelihood.keys()))

            pos_score = math.log(pos_prior)
            neg_score = math.log(neg_prior)
            for word in words:
                pos_score += pos_likelihood[word]
                neg_score += neg_likelihood[word]

            print(file + " pos: " + str(pos_score), "neg: " + str(neg_score))
            if pos_score > neg_score:
                y_pred.append("pos")

                if mode == "pos":
                    y_true.append("pos")
                    truepos += 1
                else:
                    y_true.append("neg")
                    falsepos += 1
            else:
                y_pred.append("neg")
                if mode == "pos":
                    y_true.append("pos")
                    falseneg += 1
                else:
                    y_true.append("neg")
                    trueneg += 1

    target_names = ["neg", 'pos']

    '''
    print("total: " + str(25000))
    print("processed: " + str(trueneg + truepos + falseneg + falsepos))
    print("acc: " + str((trueneg + truepos) / (trueneg + truepos + falseneg + falsepos)))
    print("precision: " + str(truepos / (truepos + falsepos)))
    print("recall: " + str(truepos / (truepos + falseneg)))
    '''
    print(classification_report(y_true, y_pred, target_names=target_names))
    return


def train_nb():
    global pos_likelihood
    global neg_likelihood
    global pos_prior
    global neg_prior
    pos_doc_count = 0
    neg_doc_count = 0
    pos_counts = {}
    neg_counts = {}

    for mode in ("neg", "pos"):
        filepath = "aclImdb/train/" + mode
        for file in sorted(os.listdir(filepath),
                           key=lambda x: (int(re.sub('\D(.*)', '', x)), x)):
            f = open(filepath + '/' + file, encoding='UTF-8')
            print(filepath + file)

            words = tokenizer(f.readline())

            if mode=="pos":
                pos_doc_count+=1
            elif mode == "neg":
                neg_doc_count+=1

            for word in words:
                if word not in pos_counts:
                    pos_counts[word] = 1
                if word not in neg_counts:
                    neg_counts[word] = 1

                if mode == "pos":
                    pos_counts[word] += 1
                elif mode == "neg":
                    neg_counts[word] += 1

            f.close()

    doc_count = pos_doc_count+neg_doc_count

    pos_prior = pos_doc_count/doc_count
    neg_prior = neg_doc_count/doc_count

    pos_likelihood = dict(map(lambda t: (t[0], math.log(t[1] / len(pos_counts))), pos_counts.items()))
    neg_likelihood = dict(map(lambda t: (t[0], math.log(t[1] / len(neg_counts))), neg_counts.items()))

    f = open("pos.csv", "w", newline="")
    w = csv.writer(f)
    for key, val in pos_likelihood.items():
        w.writerow([key, val])
    f.close()

    f = open("neg.csv", "w", newline="")
    w = csv.writer(f)
    for key, val in neg_likelihood.items():
        w.writerow([key, val])
    f.close()


if __name__ == "__main__":
    get_stop_word()
    train_nb()
    classify_nb()
