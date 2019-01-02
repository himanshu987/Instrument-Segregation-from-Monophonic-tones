import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn import neighbors
from sklearn.externals import joblib

import preprocessing


def unpack_data(data):
    src_names = np.array(map(lambda n: n[0], data))
    features = np.array(map(lambda n: n[1], data))
    print('features',features)
    labels = np.array(map(lambda n: n[2], data))
    print('labels',labels)
    return src_names, features, labels

def train_and_test(data, cv_fold=10):
    fold_unit = len(data) / cv_fold
    np.random.shuffle(data)
    accu_rates = []
    models = []
    for fold in xrange(cv_fold):             ### only one trial for now
        print 'start fold:', fold
        train_data = data[:fold_unit*fold] + data[fold_unit*(fold+1):]
        test_data = data[fold_unit*fold:fold_unit*(fold+1)]
        model = train(train_data)
        print 'training done. start testing...'
        accu_rate = test(model, test_data)
        accu_rates.append(accu_rate)
        models.append(model)
    print accu_rates
    print 'average: ', np.average(accu_rates)
    best = models[np.argmax(accu_rates)]
    save_model(best)
    return models, np.average(accu_rates)


def train(data):
    src_names, features, labels = unpack_data(data)
    print 'train feature vector dim:', features.shape
    #svc = svm.LinearSVC(C = 1.0)
    #svc = SVC(kernel = 'rbf',gamma = 0.7,C = 1.0, random_state = 0)
    svc = svm.SVC(kernel='poly', degree=3, C=1.0)
    svc.fit(features, labels)
    return svc


def predict(model, features):
    svc = model
    features=features.reshape(1,-1)#comment this line while training and uncommnet this while running demo
        predicted = svc.predict(features)
        return predicted


def test(model, data):
    src_names, features, labels = unpack_data(data)
    predicted = predict(model, features)

    # get stats for accuracy
    test_size = src_names.shape[0]
    accuracy = (predicted == labels)
    accu_rate = np.sum(accuracy) / float(test_size)
    print np.sum(accuracy), 'correct out of', test_size
    print 'accuracy rate: ', accu_rate

    # write out all the wrongly-classified samples
    wrongs = np.array([src_names, labels, predicted])
    wrongs = np.transpose(wrongs)[np.invert(accuracy)]
    with open('last_wrong.txt', 'w') as log:
        for w in wrongs:
            log.write('{} truly {} classified {}\n'.format(w[0], w[1], w[2]))
    return accu_rate


def save_model(model):
    svc = model
    joblib.dump(svc, 'last_svc.model')
    return

def load_model():
    svc = joblib.load('last_svc.model')
    return svc


def main():
    data = preprocessing.feature_extract()
    print 'processed data.'
    train_and_test(data)
    #train(data,[model_params,'svc'])
    model = load_model()
    test(model, data)

if __name__ == '__main__':
    main()
