import argparse
import boto3
import csv
import numpy as np
import optuna
import os
import pytorch_lightning as pl

import torch
import torchvision.transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import TestNet

class Objective(object):

    def __init__(self, inpath, outpath, test):
        self.inpath = inpath
        self.outpath = outpath
        self.test = test
        self.trialnum = 0

    def __call__ (self, trial):
        # Trial suggestions made in accordance with grid sampler search space,
        # not these specifications.
        b = trial.suggest_int('batch_size', 0, 156)
        c = trial.suggest_categorical('continuous', [True, False])
        d = trial.suggest_float('drop_rate', 0., 1.)
        r = trial.suggest_float('learning_rate', 0., 1.)
        w = trial.suggest_float('weight_decay', 0., 1.)

        # download data
        mnist_train = MNIST(
            root=self.inpath,
            download=True,
            transform=torchvision.transforms.ToTensor(),
            target_transform=np.float32 if c==True else None)
        
        mnist_val = MNIST(
            root=self.inpath, train=False,
            transform=torchvision.transforms.ToTensor())

        train_loader = DataLoader(
            mnist_train,
            num_workers=4,
            drop_last=True,
            batch_size=b,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True)
        
        val_loader = DataLoader(
            mnist_val,
            num_workers=2,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
            batch_size=len(mnist_val))
        
        # set up trainer
        trialpath = os.path.join(self.outpath, f"trial{self.trialnum:02}")
        self.trialnum += 1
        os.mkdir(trialpath)

        logger = pl.loggers.CSVLogger(trialpath, name="train_stats")
        callbacks = [pl.callbacks.ModelCheckpoint(
            dirpath=trialpath, filename="{epoch:03}model",
            save_top_k=-1, every_n_epochs=5)]
        if self.test:
            device_stats = pl.callbacks.DeviceStatsMonitor() 
            callbacks.append(device_stats)

        trainer = pl.Trainer(
            accelerator='gpu', devices=1, max_epochs=2 if self.test else 150,
            logger=logger, callbacks=callbacks,
            enable_progress_bar=self.test)
        
        # init and train
        model = TestNet(
            batch_size=b,
            continuous=c,
            drop_rate=d,
            learning_rate=r,
            weight_decay=w,
            epochs=10 if test else 150,
            val_batch_size=len(mnist_val))
        
        trainer.fit(model, train_loader, val_loader)
        return 0.5 # throw away value because we are not optimizing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true')
    parser.add_argument("--aws", type=lambda x: csv.DictReader(x.split("\n")))
    clargs = parser.parse_args()

    test = clargs.test
    aws_creds = next(clargs.aws)
    outpath = os.path.join(os.path.abspath(os.sep), "data", "out")
    inpath = os.path.join(os.path.abspath(os.sep), "data", "in")

    torch.set_float32_matmul_precision('high')

    session = boto3.session.Session(
        aws_access_key_id=aws_creds["Access key ID"],
        aws_secret_access_key=aws_creds["Secret access key"])

    search_space = {
        'batch_size': [16, 64],
        'drop_rate': [0.1, 0.2],
        'learning_rate': [0.0001, 0.001], # = max lr (@ epoch = maxepoch/10)
        'weight_decay': [0.3, 0.6],
        'continuous': [True, False]
    } 
    
    objective = Objective(inpath, outpath, test)
    study = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=2 if test else 32)

    # transfer files to S3
    bucket = session.resource("s3").Bucket("reg-study-bucket")

    csv_path = os.path.join(outpath, "trial_info.csv")
    study.trials_dataframe().to_csv(csv_path)
    bucket.upload_file(csv_path, "trial_info.csv")

    for t in range(2 if test else 32):
        tn = f"trial{t:02}"
        tpath = os.path.join(outpath, tn)
        bucket.upload_file(
            os.path.join(tpath, "train_stats", "version_0", "metrics.csv"),
            "metrics/{tn}.csv")
        
        for entry in os.listdir(tpath):
            if entry.split(".")[-1] == "ckpt":
                bucket.upload_file(
                    os.path.join(tpath, entry),
                    f"{tn}/epoch{entry[6:9]}.ckpt")