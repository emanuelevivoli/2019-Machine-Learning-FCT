from utils import *

def main():
    
    loader = Loader().set_train('TP1_train.tsv').set_test('TP1_test.tsv')

    train = loader.get_train()
    test = loader.get_test()

    N = (test.labels).size

    del(loader)

    X_train, y_train = train.data, train.labels
    X_test, y_test = test.data, test.labels

    del(train, test)

    svm_model = SVM()
    nb_model = NB()
    gnb_model = GaussNB()

    print("\n *** GAMMA OPTIMIZATION ... \n")

    svm_result = gamma_optimizing(svm_model, X_train, y_train)
    nb_result  = gamma_optimizing(nb_model, X_train, y_train)

    print("\n *** SAVING PLOTS ...\n")

    svm_result.save_plot()
    nb_result.save_plot()

    print("\n *** CALCULATING SVM WITH BEST GAMMA ...")

    svm_test_acc, svm_train_acc, svm_err_number, svm_err_index = \
                        svm_model.get_result(X_train, y_train, X_test, y_test, gamma=svm_result.best_gamma)
    
    print("*** CALCULATING NB WITH BEST GAMMA ...")

    nb_test_acc, nb_train_acc  , nb_err_number , nb_err_index  = \
                        nb_model.get_result(X_train, y_train, X_test, y_test, gamma=nb_result.best_gamma)

    print("*** CALCULATING GAUSSIAN NB ...\n")

    gnb_test_acc, gnb_train_acc, gnb_err_number, gnb_err_index = \
                        gnb_model.get_result(X_train, y_train, X_test, y_test)

    del(svm_model, nb_model, gnb_model)
    del(svm_result, nb_result)
    del(X_train, y_train, X_test, y_test)

    print("\n *** RESULTS ***")
    print(f'SVM test acc: {svm_test_acc} ; train acc: {svm_train_acc}')
    print(f'NB  test acc: {nb_test_acc} ; train acc: {nb_train_acc}')
    print(f'GNB test acc: {gnb_test_acc} ; train acc: {gnb_train_acc}')
    print('\n')

    results = {
        "SVM" : [svm_test_acc, svm_err_number, svm_err_index],
        "NB"  : [nb_test_acc , nb_err_number , nb_err_index],
        "GNB" : [gnb_test_acc, gnb_err_number, gnb_err_index]
    }

    print("\n *** NORMAL COMPARISON ***\n")

    normal_comparing(results, N)

    print("\n *** MCNEMAR COMPARISON ***")

    mcnemar_comparing(results)


if __name__ == "__main__":
    main()