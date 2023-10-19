import joblib
import sys
from train import preprocessed_data, print_pipe, print_shape
import time
import numpy as np

def _continue():
    if g_skip and input("You want to continue? (y/n) : ") == "n":
        exit()
    if g_skip:
        print("")

def main(subject, task, specific):
    print("\033[92mLOAD PIPELINE\033[0m")
    path = "saved/pipeline.joblib"
    print([path])
    try:
        loaded = joblib.load(path)
        print_pipe(loaded['pipeline'])
        print("Train on subject", loaded['subject'], "and task", loaded['task'])
        pipeline = loaded['pipeline'].fit(loaded['x_train'], loaded['y_train'])
    except:
        print("predict.py: Error: Loaded pipeline was not correctly saved.")
        exit()
    print("\033[92mLOAD DATA\033[0m")
    print("Subject/dataset:", subject)
    print("Task:", task)
    print("Experiment:", specific)
    x, y = preprocessed_data(subject, task, specific)
    _continue()
    while 19:
        print("\033[92mPREDICT\033[0m")
        start_time = time.time()
        num_corr = 0
        err_x = []
        err_y = []
        print(f'{"[nb]": <25}{"[prediction]": <13}{"[answer]": <13}[result]')
        for i, (_x, answer) in enumerate(zip(x, y)):
            prediction = pipeline.predict([_x])
            if prediction[0] == answer:
                correct = "CORRECT"
                num_corr += 1
            else:
                correct = "WRONG"
                err_x.append(_x)
                err_y.append(answer)
            print(f'{f"prediction {i+1:03d}:": <25}{prediction[0]: <13}{answer: <13}{correct}')
        end_time = time.time()
        print(f'\n{"Total execution time in seconds:": <33}{end_time-start_time:.4f}')
        print(f'{"Accuracy:": <33}{num_corr/len(x):.4f}')
        if len(err_x) == 0 or input("\nDo you want to drift (adapt in real time) the classifier? (y/n) : ") == "n":
            break
        print("\n\033[92mDRIFT\033[0m")
        err_x = np.array(err_x)
        err_y = np.array(err_y)
        try:
            #Because pipeline does not inherently contains partial_fit we need to call it manually
            print_shape(err_x, err_y)
            err_x = pipeline.named_steps['normalization'].fit_transform(err_x)
            err_x = pipeline.named_steps['dimensionality-reduction-algorithm'].partial_fit(err_x, err_y).transform(err_x)
            print_shape(err_x, err_y)
            pipeline.named_steps['classifier'].partial_fit(err_x, err_y, np.unique(y))
        except AttributeError:
            print("The classifier you use", pipeline.get_params()['steps'][2][1], \
                    "doesn't handle drifting. Use SGDClassifier instead.")
            exit()

if __name__ == "__main__":
    if "-s" in sys.argv:
        g_skip = False
    else:
        g_skip = True
    if "-t" in sys.argv:
        subject = input("Choose subject/dataset (1-109) : ")
        if not subject.isnumeric():
            print("predict.py: Error: Chosen subject/dataset is not numeric.")
            exit()
        subject = int(subject)
        if subject > 109 or subject < 1:
            print("predict.py: Error: Chosen subject/dataset out of range.")
            exit()
        task = input("Choose task (right-or-left-fist/fists-or-feet) : ")
        specific = input("Specific experiment? (n/1-14) : ")
        if specific != "n" and (not specific.isnumeric() or int(specific) > 14 or int(specific) < 1):
            print("predict.py: Error: Experiment value not valid.")
            exit()
    else:
        subject = 1
        task = "right-or-left-fist"
        specific = "3"
    main(subject, task, specific)
