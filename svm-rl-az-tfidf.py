# imports
import json
import warnings
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC


# função que altera o json azport antigo para gerar o json azport novo
def collapse_categories(abstracts):
    tmp_all = []
    for abstract in abstracts:
        tmp_abstract = []
        for label, f1, f2, f3, f4, f5, text in abstract:
            if label == 'A3' or label == 'D1' or label == 'D2':
                label = 'A1'
            if label == 'D7':
                label = 'D4'
            if label == 'A1':
                label = 't1'
            if label == 'A2':
                label = 't2'
            if label == 'D3':
                label = 't3'
            if label == 'D4':
                label = 't6'
            if label == 'D5':
                label = 't4'
            if label == 'D6':
                label = 't5'
            if label == 'F1':
                label = 't7'
            if label == 'F3':
                label = 't8'
            tmp_abstract.append([label, f1, f2, f3, f4, f5, text])

        tmp_all.append(tmp_abstract)

    assert (len(abstracts) == len(tmp_all))
    return tmp_all


# função que separa as labels e as features de cada sentença do json azport
def to_sentences_new(abstracts, sentences_max=None):
    sentences = []
    labels = []
    features = []
    abstracts_sentences = []
    abstracts_labels = []
    abstracts_features = []
    idents = []

    for ident, abstract in enumerate(abstracts):
        if sentences_max and len(abstract) > sentences_max:
            continue

        tmp_sentences = []
        tmp_labels = []
        tmp_features = []

        for label, f1, f2, f3, f4, f5, text in abstract:
            labels.append(label)
            sentence_features = [f1, f2, f3, f4, f5]
            features.append(sentence_features)
            sentences.append(text)

            tmp_features.append(sentence_features)
            tmp_labels.append(label)
            tmp_sentences.append(text)
            idents.append(ident)

        abstracts_sentences.append(tmp_sentences)
        abstracts_labels.append(tmp_labels)
        abstracts_features.append(tmp_features)

    assert (len(sentences) == len(labels))
    assert (len(features) == len(labels))
    assert (len(abstracts_sentences) == len(abstracts_labels))
    assert (len(abstracts_features) == len(abstracts_labels))

    return features, labels, sentences, abstracts_features, abstracts_labels, abstracts_sentences, idents


# função que calcula as listas de prev, nex e pos, além da features e labels do json azport
def abstracts_features_to_sentences(abstracts_features, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []
    abstracts_idx = []

    for i, (sentences_labels, features) in enumerate(zip(labels, abstracts_features)):
        for j, (label, feat_list) in enumerate(zip(sentences_labels, features)):
            ret.append(feat_list)
            ret_pos.append(j)
            ret_labels.append(label)
            abstracts_idx.append(i)

            if j - 1 >= 0:
                ret_prev.append(features[j - 1])
            else:
                ret_prev.append(['', '', '', '', ''])

            if j + 1 < len(features):
                ret_next.append(features[j + 1])
            else:
                ret_next.append(['', '', '', '', ''])

    return ret, ret_prev, ret_next, ret_pos, ret_labels, abstracts_idx


# função que calcula as listas de prev, nex e pos, além da sentences e labels do json azport
def abstracts_to_sentences(abstracts, labels):
    ret = []
    ret_prev = []
    ret_next = []
    ret_labels = []
    ret_pos = []
    abstracts_idx = []

    for i, (sentences_labels, sentences) in enumerate(zip(labels, abstracts)):
        for j, (label, sentence) in enumerate(zip(sentences_labels, sentences)):
            ret.append(sentence)
            ret_pos.append(j)
            ret_labels.append(label)
            abstracts_idx.append(i)

            if j - 1 >= 0:
                ret_prev.append(sentences[j - 1])
            else:
                ret_prev.append('')

            if j + 1 < len(sentences):
                ret_next.append(sentences[j + 1])
            else:
                ret_next.append('')

    return ret, ret_prev, ret_next, ret_pos, ret_labels, abstracts_idx


# função que carrega o json, chama a função to_sentences_new() e retorna o resultado
def loadFromJsonNew(file):
    data = []
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
        # altera o json para "colapsar" algumas categorias
        data = collapse_categories(data)

    return to_sentences_new(data)


# função principal
def classificador():
    # o json redacoes-az.json contém, além das features azport, os textos das sentenças
    corpus = 'redacoes-az.json'

    # parâmetros para o tfidf e chi2
    # vamor para o tamanho do n-grama
    ngrama = 1
    # valor de corte do chi2: testei com 100, 200, ..., 500 - 400 foi que deu melhor resultado
    k = 400

    # carregamento dos dados do json para as listas
    # X_sentences, X_s_prev, X_s_next contém as sentenças (texto), as sentenças anteriores e posteriores, respectivamente
    # X_pos contém as posições que as sentenças ocupam nas redações
    # Y_sentences é uma lista de listas: cada redação é uma lista de labels (uma para cada sentença)
    # X_features, X_f_prev, X_f_next contêm as features azport das sentenças, das sentenças anteriores e posteriores, respectivamente
    print("Carregando o corpus")
    _, _, _, features, labels, sentences, _ = loadFromJsonNew(corpus)
    X_features, X_f_prev, X_f_next, X_pos, Y_sentences, _ = abstracts_features_to_sentences(features, labels)
    X_sentences, X_s_prev, X_s_next, _, _, _ = abstracts_to_sentences(sentences, labels)
    # print(Y_sentences)
    # print("")

    # calculando tfidf
    print("Calculando os vetores tfidf")
    # strip_accents = ‘unicode’: Remove accents and perform other character normalization during the preprocessing step.
    # ngram_range=(1, ngrama): Define the lower and upper boundary of the range of n-values for different n-grams to be extracted.
    # All values of n such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams,
    # (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
    # stop_words=stopwords.words('portuguese'): Define that stop words in the passed list will be removed as preprocessing.
    # stopwords.words('portuguese') means the NLTK list for portuguese.
    vectorizer = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, ngrama), stop_words=stopwords.words('portuguese'))
    X_sentences = vectorizer.fit_transform(X_sentences)
    X_s_prev = vectorizer.transform(X_s_prev)
    X_s_next = vectorizer.transform(X_s_next)

    print(len(vectorizer.get_feature_names()))

    # selecionando as k melhores features tfidf com chi2
    print("Selecionando as ", k, " melhores features tfidf com chi2")
    # SelectKBest select features according to the k highest scores.
    # The first argument defines the score function, which in this case is chi2 (a statistical test)
    # k defines the number of top features to select
    selector = SelectKBest(chi2, k=k)
    X_sentences = selector.fit_transform(X_sentences, Y_sentences)
    X_s_prev = selector.transform(X_s_prev)
    X_s_next = selector.transform(X_s_next)

    # hstack do scipy - concatenação das matrizes esparsas X_sentences, X_s_prev, X_s_next
    X_sentences = hstack([X_sentences, X_s_prev, X_s_next])
    # print("sentences após hstack")
    # print(X_sentences.shape)
    # print(X_sentences)

    # hstack do numpy - concatenação das listas (dense array) X_features, X_f_prev, X_f_next
    X_features = np.hstack([X_features, X_f_prev, X_f_next])
    # binarização das features azport porque o SVM não aceita features categócias (strings)
    # enc = preprocessing.OneHotEncoder(sparse=False)
    enc = preprocessing.OneHotEncoder()
    X_features = enc.fit_transform(X_features)
    # print("features após fit_transform e hstack")
    # print(X_features.shape)
    # print(X_features)

    # ajuste do np.array X_pos para que ele tenha o mesmo número de dimensões do X_features
    X_pos = np.expand_dims(X_pos, axis=1)

    # hstack do scipy - concatenação das matrizes esparsas X_features, X_sentences e X_pos
    X_features_full = hstack([X_features, X_sentences, X_pos])
    # print("features_full após hstack")
    # print(X_features_full.shape)
    # print(X_features_full)

    # instanciação dos classificadores
    print("Treinando o SVM")
    clf = LinearSVC(dual=False, tol=1e-3)
    # print("Treinando o RL")
    # clf = LogisticRegression()

    # treino e teste do clf usando validação cruzada de 10 partições (cv = 10)
    pred = cross_val_predict(clf, X_features_full, Y_sentences, cv=10)

    # impressão dos resultados
    print("Relatorio de classificação:")
    print(classification_report(Y_sentences, pred))
    print("Matriz de confusão:")
    print(confusion_matrix(Y_sentences, pred))


# main
# método para ignorar os warnings
warnings.filterwarnings("ignore")
classificador()
