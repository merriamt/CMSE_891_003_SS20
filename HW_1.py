import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized

train_data = fetch_20newsgroups_vectorized(subset='train', remove=(), data_home=None, download_if_missing=False)
test_data = fetch_20newsgroups_vectorized(subset='test', remove=(), data_home=None, download_if_missing=False)

train_x = train_data.data.toarray()
train_x = np.hstack((np.ones((train_x.shape[0],1)),train_x))
train_y = train_data.target

test_x = test_data.data.toarray()
test_x = np.hstack((np.ones((test_x.shape[0],1)),test_x))
test_y = test_data.target

class_names = train_data.target_names

weights = np.zeros((20,test_x.shape[1]))


def update_perceptron(i, alpha):
    labels = np.where(train_y == i, 1, -1)
    for indx in range(train_x.shape[0]):
        y_hat = weights[i] @ train_x[indx]
        if np.sign(labels[indx]) != np.sign(y_hat):
            weights[i] += alpha * labels[indx] * train_x[indx]


def accuracy_report():
    report_accuracies = [[0,0] for x in range(20)]
    scores = np.zeros(20)
    total = 0
    for i in range(test_x.shape[0]):
        for indx in range(weights.shape[0]):
            scores[indx] = weights[indx] @ test_x[i]
        report_accuracies[test_y[i]][1] += 1
        if np.amax(np.argmax(scores)) == test_y[i]:
            report_accuracies[test_y[i]][0] += 1
            total += 1
    for i in range(20):
        print("The ", class_names[i], " paper was correctly identified ", report_accuracies[i][0], " out of",
               report_accuracies[i][1], " time for a label accuracy of ",
              report_accuracies[i][0]/report_accuracies[i][1] * 100, " percent.")
    print("\n The total accuracy was ", total/test_x.shape[0], " percent.")


def learn(alpha, max_iter):
    for i in range(20):
        for x in range(max_iter):
            update_perceptron(i, alpha)
        print("Learning is ", 5 * (i+1), " percent complete.")

learn(0.02, 50)
accuracy_report()

