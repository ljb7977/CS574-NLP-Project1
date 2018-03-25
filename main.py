import math
import nltk
import os
import re
import csv

vocab = set()
word_count = 0

'''
def train_nb(training_documents, mode):
    global pos_allcount, neg_allcount
    global pos_likelihood, neg_likelihood

    for doc in training_documents:
        t = doc.split()
        score = int(t[0])
        for word in t[1:]:
            index, count = word.split(':')
            # print('index: '+index)
            if mode == "pos":
                if pos_counts[int(index)] == 0:
                    pos_counts[int(index)] = 1
                pos_counts[int(index)] += int(count)
                pos_allcount += int(count)
            elif mode == "neg":
                if neg_counts[int(index)] == 0:
                    neg_counts[int(index)] = 1
                neg_counts[int(index)] += int(count)
                neg_allcount += int(count)
            # print(score, pos_counts)

    if mode == "pos":
        for c in pos_counts:
            if c == 0:
                print(str(c) + "is zero")
        pos_likelihood = list(map(lambda t: t / pos_allcount, pos_counts))
        print(pos_likelihood)
        print("all counts: " + str(pos_allcount))
    elif mode == "neg":
        for c in neg_counts:
            if c == 0:
                print(str(c) + "is zero")
        neg_likelihood = list(map(lambda t: t / neg_allcount, neg_counts))
        print(neg_likelihood)
        print("all counts: " + str(neg_allcount))

    return
'''

def classify_nb():
    truepos = 0
    trueneg = 0
    falsepos = 0
    falseneg = 0
    pos_likelihood = {}
    neg_likelihood = {}

    f = open("pos.csv")
    for key, val in csv.reader(f):
        pos_likelihood[key] = float(val)
    f.close()

    f = open("neg.csv")
    for key, val in csv.reader(f):
        neg_likelihood[key] = float(val)
    f.close()

    for mode in ("neg", "pos"):
        filepath = "aclImdb/test/" + mode + "/"
        for file in sorted(os.listdir(filepath),
               key=lambda x: (int(re.sub('\D(.*)', '', x)), x)):
            document = open(filepath+file, encoding='UTF-8')

            words = re.sub("(<.*?>|[^a-zA-Z'])+", ' ', document.readline()).lower()
            words = words.split()
            #words = document.read().split()
            new_words = []
            for word in words:
                if word in pos_likelihood.keys():
                    new_words.append(word)

            pos_score = 0
            neg_score = 0
            for word in new_words:
                pos_score += pos_likelihood[word]
                neg_score += neg_likelihood[word]

            print('score')
            print(pos_score, neg_score)
            if(pos_score > neg_score):
                if mode == "pos":
                    truepos+=1
                else:
                    falsepos+=1
            else:
                if mode == "pos":
                    falseneg+=1
                else:
                    trueneg+=1

    print("acc: "+str((trueneg+truepos)/(trueneg+truepos+falseneg+falsepos)))
    print("precision: "+str(truepos/(truepos+falsepos)))
    print("recall: "+str(truepos/(truepos+falseneg)))
    return


def train_nb():
    pos_counts = {}
    neg_counts = {}

    global vocab

    vocab_f = open('aclImdb/imdb.vocab')
    word_list = set(vocab_f.read().splitlines())
    vocab_f.close()

    for mode in ("neg", "pos"):
        filepath = "aclImdb/train/" + mode
        for file in sorted(os.listdir(filepath),
                           key=lambda x: (int(re.sub('\D(.*)', '', x)), x)):
            f = open(filepath + '/' + file, encoding='UTF-8')
            print(filepath+file)
            words = re.sub("(<.*?>|[^a-zA-Z'])+", ' ', f.readline()).lower()
            words = words.split()
            for word in words:
                if word not in pos_counts:
                    pos_counts[word] = 1
                if word not in neg_counts:
                    neg_counts[word] = 1

                if mode == "pos":
                    pos_counts[word] += 1
                elif mode == "neg":
                    neg_counts[word] += 1

            # print(words)
            vocab.update(words)
            f.close()

    pos_likelihood = dict(map(lambda t: (t[0], math.log(t[1]/ len(pos_counts))), pos_counts.items()))
    neg_likelihood = dict(map(lambda t: (t[0], math.log(t[1]/ len(neg_counts))), neg_counts.items()))

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
    #print(vocab)
    #
    # print(len(vocab))
    # print(len(neg_counts))
    # print(len(pos_counts))
    # print(pos_likelihood)
    # print(neg_likelihood)
    #print(vocab - word_list)

if __name__ == "__main__":
    #train_nb()
    classify_nb()

