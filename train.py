from glob import glob
import numpy as np
from keras.models import load_model
from model import *
from data import *
import json
import sys
from sklearn.utils import class_weight
import keras
from metrics import dice_coef, dice_coef_loss
import configparser
import os


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

    # parse config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    model_str = "{}_v{}".format(config['model']['name'], config['model']['ver'])
    model_dir = "models/{}".format(model_str)
    lr = 1e-4
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
        model = unet()
        # save sumamry printout of model to directory
        og_stdout = sys.stdout
        with open("{}/{}_summary.txt", "w") as f:
            sys.stdout = f
            print(model.summary())
            sys.stdout = og_stdout
    else:
        model = load_model("{}/{}".format(model_dir, model_str),
                           custom_objects={"dice_coef": dice_coef,
                                           "dice_coef_loss": dice_coef_loss})
        model.load_weights("{}/{}_w".format(model_dir, model_str))
        model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    # parse config
    eps = int(config['training']['epochs'])
    bs = int(config['training']['batch_size'])
    vs = float(config['training']['validation_size'])
    start = int(config['training']['start'])
    end = int(config['training']['end'])
    interval = int(config['training']['interval'])

    # write details to log file
    log = open("{}/log.txt", "a+")
    log.write("Model: {}_v{}".format(config['model']['name'],config['model']['ver']))
    log.write("Epochs: {}".format(epochs))
    log.write("Batch Size: {}".format(bs))
    log.write("Validation split: {}".format(vs))
    log.write("Starting patient no: {}".format(start))
    log.write("Ending partient no: {}".format(end))
    log.write("Interval:{}".format(interval))

    # train on batches of patients with size __interval__
    # model and weights are saved at each interval
    cur_start = start
    while cur_start < end
        cur_end = min(cur_start+interval, end)
        x,y = generate_data(cur_start, cur_end)
        model.fit(x,y, epohs=eps, batch_size=bs, validation_size=vs, shuffle=True)
        model.save("{}/{}".format(model_dir, model_str))
        model.save_weights("{}/{}_w".format(model_dir, model_str))
        cur_start += interval

