# imports
import json
import warnings
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
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
def load_from_json_new(file):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
        # altera o json para "colapsar" algumas categorias
        data = collapse_categories(data)

    return to_sentences_new(data)


# ********* funções Ana *********
# função tokens
def num_tokens(lista_s):
    tam = len(lista_s)
    tokens = []
    numtokens = []

    for i in range(tam):
        aux = word_tokenize(lista_s[i])
        tokens.append(aux)
        numtokens.append(len(aux))

    return numtokens


# função pronome de 1a pessoa
def tem_1a_pessoa(lista_s):
    tam = len(lista_s)
    ppyn = []

    for i in range(tam):
        if re.search('\\beu\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmeu\\b', lista_s[i], re.IGNORECASE) or \
                re.search('\\bnós\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnosso\\b', lista_s[i],
                                                                               re.IGNORECASE) or \
                re.search('\\bme\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmim\\b', lista_s[i], re.IGNORECASE) \
                or re.search('\\bcomigo\\b', lista_s[i], re.IGNORECASE) or re.search('\\bminha\\b', lista_s[i],
                                                                                     re.IGNORECASE) or \
                re.search('\\bminhas\\b', lista_s[i], re.IGNORECASE) or re.search('\\bmeus\\b', lista_s[i],
                                                                                  re.IGNORECASE) or \
                re.search('\\bnossa\\b', lista_s[i], re.IGNORECASE) or re.search('\\bnossos\\b', lista_s[i],
                                                                                 re.IGNORECASE) or \
                re.search('\\bnossass\\b', lista_s[i], re.IGNORECASE):
            ppyn.append(1)
        else:
            ppyn.append(0)

    return ppyn

# *******************************

# função que prepara os atributos tf-idf
def atributos_tfidf(n_grama, k_chi):
    # o json redacoes-az.json contém, além das features azport, os textos das sentenças
    corpus = 'redacoes-az.json'

    # carregamento dos dados do json para as listas X_sentences, X_s_prev, X_s_next contém as sentenças (texto),
    # as sentenças anteriores e posteriores, respectivamente X_pos contém as posições que as sentenças ocupam nas
    # redações Y_sentences é uma lista de listas: cada redação é uma lista de labels (uma para cada sentença)
    # X_features, X_f_prev, X_f_next contêm as features azport das sentenças, das sentenças anteriores e posteriores,
    # respectivamente
    # print("Carregando o corpus")
    _, _, _, features, labels, sentences, _ = load_from_json_new(corpus)
    X_features, X_f_prev, X_f_next, X_pos, Y_sentences, _ = abstracts_features_to_sentences(features, labels)
    X_sentences, X_s_prev, X_s_next, _, _, _ = abstracts_to_sentences(sentences, labels)

    # print("Calculando os vetores tfidf")
    vectorizer = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, n_grama),
                                     stop_words=stopwords.words('portuguese'))
    X_sentences = vectorizer.fit_transform(X_sentences)
    X_s_prev = vectorizer.transform(X_s_prev)
    X_s_next = vectorizer.transform(X_s_next)

    # print(len(vectorizer.get_feature_names()))

    # selecionando as k melhores features tfidf com chi2
    # print("Selecionando as ", k_chi, " melhores features tfidf com chi2")
    selector = SelectKBest(chi2, k=k_chi)
    X_sentences = selector.fit_transform(X_sentences, Y_sentences)
    X_s_prev = selector.transform(X_s_prev)
    X_s_next = selector.transform(X_s_next)

    # hstack do scipy - concatenação das matrizes esparsas X_sentences, X_s_prev, X_s_next
    X_features_full = hstack([X_sentences, X_s_prev, X_s_next])

    return X_features_full, Y_sentences


# função que prepara os atributos azport. Se proj_ana=True, os dois atributos do projeto da Ana são adicionados
def atributos_azport(proj_ana=False):
    # o json redacoes-az.json contém, além das features azport, os textos das sentenças
    corpus = 'redacoes-az.json'

    # carregamento dos dados do json para as listas X_sentences, X_s_prev, X_s_next contém as sentenças (texto),
    # as sentenças anteriores e posteriores, respectivamente X_pos contém as posições que as sentenças ocupam nas
    # redações Y_sentences é uma lista de listas: cada redação é uma lista de labels (uma para cada sentença)
    # X_features, X_f_prev, X_f_next contêm as features azport das sentenças, das sentenças anteriores e posteriores,
    # respectivamente
    # print("Carregando o corpus")
    _, _, _, features, labels, sentences, _ = load_from_json_new(corpus)
    X_features, X_f_prev, X_f_next, X_pos, Y_sentences, _ = abstracts_features_to_sentences(features, labels)
    X_sentences, X_s_prev, X_s_next, _, _, _ = abstracts_to_sentences(sentences, labels)

    # hstack do numpy - concatenação das listas (dense array) X_features, X_f_prev, X_f_next
    X_features = np.hstack([X_features, X_f_prev, X_f_next])
    # binarização das features azport porque o SVM não aceita features categócias (strings)
    enc = preprocessing.OneHotEncoder()
    X_features = enc.fit_transform(X_features)

    # ajuste dos np.array X_pos, X_numtok e X_pro1ap para que eles tenham o mesmo número de dimensões do X_features
    X_pos = np.expand_dims(X_pos, axis=1)

    if proj_ana:
        # ******* Features Ana **********
        # X_numtok contém os tamanhos em tokens das sentenças
        X_numtok = num_tokens(X_sentences)
        # X_numtok contém valores 0 ou 1 sinalizando a presença ou não de pronomes de 1a pessoa
        X_pro1ap = tem_1a_pessoa(X_sentences)
        # *******************************

        # ajuste dos np.array X_numtok e X_pro1ap para que eles tenham o mesmo número de dimensões do X_features
        X_numtok = np.expand_dims(X_numtok, axis=1)
        X_pro1ap = np.expand_dims(X_pro1ap, axis=1)

        # hstack do scipy - concatenação das matrizes esparsas X_features, X_sentences, X_pos, X_numtok e X_pro1ap
        X_features_full = hstack([X_features, X_pos, X_numtok, X_pro1ap])
    else:
        X_features_full = hstack([X_features, X_pos])

    return X_features_full, Y_sentences


# função que treina e testa o classificador, e imprime o relatório de resultados
def classificador(X_features, Y_sentences):

    # instanciação do classificador SVM
    clf = LinearSVC(dual=False, tol=1e-3)

    # treino e teste do clf usando validação cruzada de 10 partições (cv = 10)
    pred = cross_val_predict(clf, X_features, Y_sentences, cv=10)

    # impressão dos resultados
    print("Relatorio de classificação:")
    print(classification_report(Y_sentences, pred))
    print("Matriz de confusão:")
    print(confusion_matrix(Y_sentences, pred))
    print()


# main
warnings.filterwarnings("ignore")

# parâmetros a serem testados com tf-idf
n_gramas = [1, 2, 3]
ks_chi = [100, 200, 300, 400, 500, 1000]

print("RESULTADOS USANDO OS ATRIBUTOS AZPORT")
X_az, y = atributos_azport()
classificador(X_az, y)
print()

print("RESULTADOS USANDO OS ATRIBUTOS AZPORT + ATRIBUTOS ANA")
X_az, y = atributos_azport(proj_ana=True)
classificador(X_az, y)
print()

print("RESULTADOS USANDO OS ATRIBUTOS TF-IDF")
for i in n_gramas:
    for j in ks_chi:
        print("UTILIZANDO n-grama = %i e k_chi2 = %i" % (i, j))
        X_tfidf, y = atributos_tfidf(i, j)
        classificador(X_tfidf, y)
        print()

print("RESULTADOS USANDO OS ATRIBUTOS AZPORT + ANA + TF-IDF")
X_az, y = atributos_azport(proj_ana=True)
# ajustar os parâmetros da função com os melhores valores
X_tfidf, _ = atributos_tfidf(1, 1000)
# hstack do scipy - concatenação das matrizes X_az, X_tfidf
X_full = hstack([X_az, X_tfidf])
classificador(X_full, y)
print()