from utils import *
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--hyper-opt', 
                    default="",
                    help='[default ""] It is the path of the solver bonmin. It allows to use hyperparameters optimization for C and gamma (SVM model)')

parser.add_argument('--csv-log', 
                    default=False,
                    action='store_true',
                    help='[default False] Flag to log in csv result from hyperparameters optimization')

parser.add_argument('--opti-steps', 
                    default=25,
                    help='[default 25] number of steps in optimization process')

parser.add_argument('--seed', 
                    default=None,
                    help='[default False] is the seed for the randomize processes (shuffle)')

parser.add_argument('--n-split', 
                    default=5,
                    help='[default 5] number of split in k-fold cross validation')

args = parser.parse_args()

with_hyper_optimization = bool(args.hyper_opt)

if with_hyper_optimization:
    set_solver_path(args.hyper_opt if bool(args.hyper_opt) else bool(args.hyper_opt))
    set_opti_steps(int(args.opti_steps))

set_seed(args.seed)
set_split(int(args.n_split))
set_csv(args.csv_log)

def main():
    
    loader = Loader().set_train('TP1_train.tsv').set_test('TP1_test.tsv')

    train = loader.get_train()
    test = loader.get_test()

    N = (test.labels).size

    # del(loader)

    X_train, y_train = train.data, train.labels
    X_test, y_test = test.data, test.labels

    # del(train, test)

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


    print(f'\n *** CALCULATING SVM ...')

    svm_train_acc, svm_test_acc, svm_err_number, svm_err_index = \
                        svm_model.get_result(X_train, y_train, X_test, y_test, params=[svm_result.best_param, 1])

    print(f'train acc: {svm_train_acc} \ntest acc: {svm_test_acc} \nerror numb: {svm_err_number}')
    
    if with_hyper_optimization:
        print(f'\n *** CALCULATING HYP SVM ...')

        hyp_svm_train_acc, hyp_svm_test_acc, hyp_svm_err_number, hyp_svm_err_index = \
                            hype_svm_model.get_result(X_train, y_train, X_test, y_test, params=hyper_params)

        print(f'train acc: {hyp_svm_train_acc} \ntest acc: {hyp_svm_test_acc} \nerror numb: {hyp_svm_err_number}')


    print(f'\n *** CALCULATING NB ...')

    nb_train_acc, nb_test_acc, nb_err_number , nb_err_index  = \
                        nb_model.get_result(X_train, y_train, X_test, y_test, params=[nb_result.best_param, 1])

    print(f'train acc: {nb_train_acc} \ntest acc: {nb_test_acc} \nerror numb: {nb_err_number}')

    print("\n *** CALCULATING GAUSSIAN NB ...")

    gnb_train_acc, gnb_test_acc, gnb_err_number, gnb_err_index = \
                        gnb_model.get_result(X_train, y_train, X_test, y_test)

    print(f'train acc: {gnb_train_acc} \ntest acc: {gnb_test_acc} \nerror numb: {gnb_err_number}')

    # del(svm_model, nb_model, gnb_model)
    # del(svm_result, nb_result)

    # del(X_train, y_train, X_test, y_test)

    print("\n\n *** RESULTS ***")
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