from utils import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--hyper-opt', 
                    default=False,
                    action='store_true',
                    help='[default False] it allow to use hyperparameters optimization for C and gamma (SVM model)')

args = parser.parse_args()

with_hyper_optimization = args.hyper_opt

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
    if with_hyper_optimization: 
        hype_svm_model = SVM()
    nb_model = NB()
    gnb_model = GaussNB()

    print("\n *** PARAM OPTIMIZATION ... \n")

    svm_result = param_optimizing(svm_model, X_train, y_train)
    nb_result  = param_optimizing(nb_model, X_train, y_train)

    print("\n *** SAVING PLOTS ...\n")

    svm_result.save_plot()
    nb_result.save_plot()

    if with_hyper_optimization:
        hyper_params = hyperparameter_optimization(X_train, y_train, X_test , y_test)


    print("\n *** BEST PARAMS ***")
    print(f'SVM      best gamma: {svm_result.best_param} , default C : 1 ') 
    if with_hyper_optimization:
        print(f'hyperSVM best gamma: {hyper_params[0]} ,  best   C : {hyper_params[1]} ') 
    print(f'NB       best bandwidth: {nb_result.best_param} ')
    print(f'GNB      no params ') # ; train err: {1 - gnb_train_acc}')
    print('\n')


    print(f' *** CALCULATING SVM ...')

    svm_test_acc, svm_train_acc, svm_err_number, svm_err_index = \
                        svm_model.get_result(X_train, y_train, X_test, y_test, params=[svm_result.best_param, 1])

    if with_hyper_optimization:
        print(f' *** CALCULATING HYP SVM ...')

        hyp_svm_test_acc, hyp_svm_train_acc, hyp_svm_err_number, hyp_svm_err_index = \
                            hype_svm_model.get_result(X_train, y_train, X_test, y_test, params=hyper_params)

    
    print(f' *** CALCULATING NB ...')

    nb_test_acc, nb_train_acc  , nb_err_number , nb_err_index  = \
                        nb_model.get_result(X_train, y_train, X_test, y_test, params=[nb_result.best_param, 1])

    print(" *** CALCULATING GAUSSIAN NB ...\n")

    gnb_test_acc, gnb_train_acc, gnb_err_number, gnb_err_index = \
                        gnb_model.get_result(X_train, y_train, X_test, y_test)

    del(svm_model, nb_model, gnb_model)
    del(svm_result, nb_result)

    del(X_train, y_train, X_test, y_test)

    print("\n *** RESULTS ***")
    print(f'SVM      test err: {100*(1 - svm_test_acc)} % ') # ; train err: {1 - svm_train_acc}')

    if with_hyper_optimization:
        print(f'hyperSVM test err: {100*(1 - hyp_svm_test_acc)} % ') # ;train err: {1 - hyp_svm_train_acc}')

    print(f'NB       test err: {100*(1 - nb_test_acc)} % ') # ;train err: {1 - nb_train_acc}')
    print(f'GNB      test err: {100*(1 - gnb_test_acc)} % ') # ; train err: {1 - gnb_train_acc}')
    print('\n')

    results = {
        "SVM"    : [svm_test_acc    , svm_err_number    , svm_err_index],
        "NB"     : [nb_test_acc     , nb_err_number     , nb_err_index],
        "GNB"    : [gnb_test_acc    , gnb_err_number    , gnb_err_index]
    }

    if with_hyper_optimization:
        results["hypSVM"] = [hyp_svm_test_acc, hyp_svm_err_number, hyp_svm_err_index]

    print("\n *** NORMAL COMPARISON ***\n")

    normal_comparing(results, N)

    print("\n *** MCNEMAR COMPARISON ***")

    mcnemar_comparing(results)


if __name__ == "__main__":
    main()