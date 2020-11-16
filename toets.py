import keras_face_vericator
import os
import time
import itertools
import matplotlib.pyplot as plt
import random
import concurrent.futures
import test_metrics
import json
import opencv_face_verificator

dataDir = "FERET_data"

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

def keras_toetser(test_data_paths, model):
    '''
    Tests a verifier that requires 2 images
    input:
    return: json containing confidences array, ground truths, and run time
    '''

    y_score = []
    y_target = []
    run_time = []

    for path_set in progressBar(test_data_paths, prefix='Progress:', suffix='Complete', length=50):
        image1 = path_set[0]
        image2 = path_set[1]

        t0 = time.time()
        score = keras_face_vericator.validate(image1, image2, model)
        elapsed_time = time.time() - t0
        run_time.append(elapsed_time)
        y_score.append(score)
        if os.path.basename(image1)[0:6] != os.path.basename(image2)[0:6]:
            y_target.append(0)

        else:
            y_target.append(1)

    # for path_set in test_data_paths:
    #     image1 = path_set[0]
    #     image2 = path_set[1]
    #
    #     t0 = time.time()
    #     score = keras_face_vericator.validate(image1, image2, model)[0][0]
    #     elapsed_time = time.time() - t0
    #     print(elapsed_time)
    #     run_time.append(elapsed_time)
    #     y_score.append(score)
    #     if os.path.basename(image1)[0:6] != os.path.basename(image2)[0:6]:
    #         y_target.append(0)
    #
    #     else:
    #         y_target.append(1)

    return y_score, y_target, run_time

def cv_toetser(test_data_paths):
    '''
    Tests a verifier that requires 2 images
    input:
    return: json containing confidences array, ground truths, and run time
    '''

    y_score = []
    y_target = []
    run_time = []

    for path_set in progressBar(test_data_paths, prefix='Progress:', suffix='Complete', length=50):
        image1 = path_set[0]
        image2 = path_set[1]

        t0 = time.time()
        score = opencv_face_verificator.validate(image1, image2)
        elapsed_time = time.time() - t0
        run_time.append(elapsed_time)
        y_score.append(score)
        if os.path.basename(image1)[0:6] != os.path.basename(image2)[0:6]:
            y_target.append(0)
        else:
            y_target.append(1)

    return y_score, y_target, run_time



def toets(path_set):
    model = keras_face_vericator.senet50_model
    image1 = path_set[0]
    image2 = path_set[1]
    t0 = time.time()
    score = keras_face_vericator.validate(image1, image2, model)
    elapsed_time = time.time() - t0
    if os.path.basename(image1)[0:6] != os.path.basename(image2)[0:6]:
        target = 0
    else:
        target = 1
    print('score :' + str(score))
    print('elapsed time :' + str(elapsed_time))
    return [score, target, elapsed_time]

def image_count_summary():
    print('number of candidates in total: ' + str(len(os.listdir(dataDir))))
    for n in range(20):
        largerFolder = []
        for ids in os.listdir(dataDir):
            if len(os.listdir(os.path.join(dataDir, ids))) > n:
                largerFolder.append(ids)
        print('candidates with more than ' + str(n) + ' images : ' + str(len(largerFolder)))

def data_histogram() :
    img_cnt = []
    for ids in os.listdir(dataDir):
        id_path = os.path.join( dataDir, ids)
        img_cnt.append(len(os.listdir(id_path)))

    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    axs.hist(img_cnt, bins=50)
    plt.show()




def create_test_combinations(minPhotos=5, maxPhotos=-1, balanceData=True):
    candidates_to_use = []
    # create a list of usable candidates based on how many photos we want to use if maxPhotos== -1 the no max number of photos is required
    if maxPhotos == -1:
        for ids in os.listdir(dataDir):
            if len(os.listdir(os.path.join(dataDir, ids))) >= minPhotos:
                candidates_to_use.append(os.path.join(dataDir, ids))
                used_candidates = candidates_to_use
    else:
        for ids in os.listdir(dataDir):
            if len(os.listdir(os.path.join(dataDir, ids))) >= minPhotos and len(os.listdir(os.path.join(dataDir, ids))) <= maxPhotos:
                candidates_to_use.append(os.path.join(dataDir, ids))
                used_candidates = candidates_to_use

    print(len(candidates_to_use))
    positive_test_data_paths = []
    all_negative_test_data_paths = []

    for candidate in candidates_to_use :
        candidate_images = os.listdir(candidate)
        candidate_images = [os.path.join(candidate, i) for i in candidate_images]
        # create the positive test combinations for this ID
        id_combinations = list(itertools.combinations(candidate_images, 2))
        batch_length = len(id_combinations)
        positive_test_data_paths = positive_test_data_paths + id_combinations

        # create the imposter test combinations for this ID
        candidates_to_use.remove(candidate)
        for candidate_image in candidate_images:
            for remaining_candidate in candidates_to_use:
                thingy = [(candidate_image, os.path.join(remaining_candidate, remaining_candidate_image)) for remaining_candidate_image in os.listdir(remaining_candidate)]
                all_negative_test_data_paths = all_negative_test_data_paths + thingy

    if balanceData == True :
        posi_length = len(positive_test_data_paths)
        negative_test_data_paths = random.sample(all_negative_test_data_paths, posi_length)
    else:
        negative_test_data_paths = all_negative_test_data_paths

    print(len(positive_test_data_paths))
    print(len(set(positive_test_data_paths)))
    print(len(negative_test_data_paths))
    print(len(set(negative_test_data_paths)))

    return positive_test_data_paths, negative_test_data_paths, used_candidates


if __name__ == '__main__' :

    #positive_data, imposter_data, candidates = create_test_combinations(8, 8, balanceData=True)
    #full_data = positive_data + imposter_data
    model = keras_face_vericator.resnet50_model


    filename = 'result1604625071.json'
    with open(filename, 'r') as f:
        data = json.load(f)

    full_data = data['test_data']
    candidates = data['candidates']

    y_score, y_target, runtime = keras_toetser(full_data, model)
    #y_score, y_target, runtime = cv_toetser(full_data)

    print(y_score)
    print(y_target)
    print(runtime)

    results = {}

    results['candidates'] = candidates
    results['test_data'] = full_data
    results['model'] = 'resnet50'
    results['y_score'] = y_score
    results['y_target'] = y_target
    results['runtimes'] = runtime

    precision = test_metrics.precision(y_score, y_target)
    recall = test_metrics.recall(y_score, y_target)
    acc = test_metrics.accuracy(y_score, y_target)
    print(precision)
    print(recall)

    results['precision'] = precision
    results['recall'] = recall
    results['accuracy'] = acc

    filename = 'result' + str(int(time.time())) + '_' + results['model'] + '_' + '.json'
    with open(filename, 'w') as f:
        json.dump(results, f)

