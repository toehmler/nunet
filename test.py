import os
import csv
import configparser
from model import *


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('config.ini')

    # load model to test
    model_str = "{}/_v{}".format(cofig['model']['name'], config['model']['ver'])
    model_dir = "models/{}".format(model_str)
    model = load_model("{}/{}".format(model_dir, model_str),
                       custom_objects={"dice_coef": dice_coef,
                                       "dice_coef_loss": dice_coef_loss})
    model.load_weights("{}/{}_w".format(model_dir, model_str))
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])

    # open file to write test results to

    # save testing outputs to csv




