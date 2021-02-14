import os
import csv
import json
import configparser
from tqdm import tqdm
from keras.models import load_model
import csv
from model import *
from metrics import dice_coef, dice_coef_loss
import datetime
from glob import glob
from metrics import *
from data import *

# load config (used to get path to data and other model specifics)
with open('config.json') as config_file:
    jconfig = json.load(config_file)

lr = 1e-4

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')

    # load model to test
    model_str = "{}_v{}".format(config['model']['name'], config['model']['ver'])
    model_dir = "models/{}".format(model_str)
    model = load_model("{}/{}".format(model_dir, model_str),
                       custom_objects={"dice_coef": dice_coef,
                                       "dice_coef_loss": dice_coef_loss})
    model.load_weights("{}/{}_w".format(model_dir, model_str))
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    # open file to write test results to
    result_filename = "{}/{}_{}.csv".format(model_dir, model_str,
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(result_filename, mode="a+") as result_file:
        result_writer = csv.writer(result_file, delimiter=',',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(['dice_whole','dice_enhancing','dice_core',
                                'sen_whole','sen_enhancing','sen_score',
                                'spec_whole','spec_enhancing','spec_core'])


    # parse config file
    start = int(config['testing']['start'])
    end = int(config['testing']['end'])
    current = start
    pbar = tqdm(total = (end - start))
    dw = 0
    de = 0
    dc = 0
    sn_w = 0
    sn_e = 0
    sn_c = 0
    sp_w = 0
    sp_e = 0
    sp_c = 0
    while current < end:
        path = glob(jconfig['root'] + '/*pat{}*'.format(current))[0]
        scans = load_scans(path)
        scans = norm_scans(scans)
        gt = []
        pred = []
        for slice_no in range(scans.shape[0]):
            test_slice = scans[slice_no:slice_no+1,:,:,:4]
            test_label = scans[slice_no:slice_no+1,:,:,4]
            prediction = model.predict(test_slice, batch_size=32)
            prediction = prediction[0]
            prediction = np.around(prediction)
            prediction = np.argmax(prediction, axis=-1)
            gt.extend(test_label[0])
            pred.extend(prediction)
        gt = np.array(gt)
        pred = np.array(pred)
        dice_whole = DSC_whole(pred, gt)
        dice_en = DSC_en(pred, gt)
        dice_core = DSC_core(pred, gt)

        sen_whole = sensitivity_whole(pred, gt)
        sen_en = sensitivity_en(pred, gt)
        sen_core = sensitivity_core(pred, gt)

        spec_whole = specificity_whole(pred, gt)
        spec_en = specificity_en(pred, gt)
        spec_core = specificity_core(pred, gt)

        with open(result_filename, mode="a+") as result_file:
            result_writer = csv.writer(result_file, delimiter=',',
                                       quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow([dice_whole,dice_en,dice_core,
                                   sen_whole,sen_en,sen_core,
                                   spec_whole,spec_en,spec_core])


        dw += dice_whole
        de += dice_en
        dc += dice_core
        sn_w += sen_whole
        sn_e += sen_en
        sn_c += sen_core
        sp_w += spec_whole
        sp_e += spec_en
        sp_c += spec_core
        current += 1
        pbar.update(1)

    # write average results to csv
    count = end - start
    result_writer.writerow(["{:0.4f}".format(dw/count),
                            "{:0.4f}".format(de/count),
                            "{:0.4f}".format(dc/count),
                            "{:0.4f}".format(sn_w/count),
                            "{:0.4f}".format(sn_e/count),
                            "{:0.4f}".format(sn_c/count),
                            "{:0.4f}".format(sp_w/count),
                            "{:0.4f}".format(sp_e/count),
                            "{:0.4f}".format(sp_c/count)])
    pbar.close()






