from helpers import *
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = []
models.append(("Gaussian Naive Bayes", GaussianNB()))
models.append(("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)))
# models.append(("SVM",svm.SVC(kernel='linear', C = 1)))
models.append(("MLP ", MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=1e-4, solver='adam', tol=1e-5, random_state=1,
                    learning_rate_init=.001)))

eq = []
eq.append(("Original", 0))
eq.append(("HE", 1))
eq.append(("DHE", 2))
eq.append(("FUSION", 3))

print("\n\n\n-----Imports finished-----")

size_face = (20, 20)
# print("\n\nReading Face Dataset")
# X1, y1 = read_face(size_face)
# run_models(X1, y1, models)

# print("\n\nReading Face Dataset (Enhanced)")
# for n, flag in eq:
#     print("\n{}".format(n))
#     X1, y1 = read_face(size_face, flag)
#     run_models(X1, y1, models)


size_exdark = (30, 30)
# print("\n\nReading ExDark Dataset")
# X2, y2 = read_exdark(size_exdark)
# run_models(X2, y2, models)

print("\n\nReading ExDark Dataset (Enhanced)")
for n, flag in eq:
    print("\n{}".format(n))
    X2, y2 = read_exdark(size_exdark, flag)
    run_models(X2, y2, models)
