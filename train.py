from glob import glob
import numpy as np
from keras.models import load_model
from model import *
from data import *
import json
import sys
from sklearn.utils import class_weight
import keras

# load config (used to get path to data and other model specifics)
with open('config.json') as config_file:
    config = json.load(config_file)

''' generate_data: generates normalized and shuffled data to train with
@param: start the starting index of the patient to generate data from
@param: end the ending index of the patient to generate data from
@returns: np array with all scans shuffled and normalized '''
def generate_data(start, end):
    current = start
    x = []
    y = []
    pbar = tqdm(total = (end - start))
    while current < end:
        path = glob(config['root'] + '/*pat{}*'.format(current))[0]
        scans = load_scans(path)
        scans = norm_scans(scans)
        current_x = []
        labels = []
        for slice in scans:
            slice_label = slice[:,:,4]
            slice_x = slice[:,:,:4]
            current_x.append(slice_x)
            categorical = keras.utils.to_categorical(slice_label, num_classes=5)
            labels.append(categorical)

        current_x = np.array(current_x)
        labels = np.array(labels)
        x.extend(current_x)
        y.extend(labels)
        current += 1
        pbar.update(1)

    shuffle = list(zip(x, y))
    np.random.shuffle(shuffle)
    x, y = zip(*shuffle)
    x = np.array(x)
    y = np.array(y)
    pbar.close()
    return x,y

if __name__ == "__main__":
    model_name = sys.argv[1]
    start_pat = int(sys.argv[2])
    end_pat = int(sys.argv[3])
    eps = int(sys.argv[4])
    bs = int(sys.argv[5])
    vs = float(sys.argv[6])
    arch_path = 'models/architectures/test_model.json'
    with open(arch_path, 'r') as json_file:
        arch = json.load(json_file)
        unet_model = keras.models.model_from_json(json.dumps(arch))

#    unet_model = keras.models.model_from_json('models/architectures/test_model.json')
    unet_model.load_weights('models/weights/test_model_weights')





