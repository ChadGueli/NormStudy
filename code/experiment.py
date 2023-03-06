import argparse
import boto3
import numpy as np
import optuna
import os
import pytorch_lightning as pl

import pandas as pd
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
        os.mkdir(trialpath)

        logger = pl.loggers.CSVLogger(trialpath, name="train_stats")
        callbacks = [pl.callbacks.ModelCheckpoint(
            dirpath=trialpath, filename="{epoch:03}model",
            save_top_k=-1, every_n_epochs=5)]
        if self.test:
            device_stats = pl.callbacks.DeviceStatsMonitor() 
            callbacks.append(device_stats)

        trainer = pl.Trainer(
            accelerator='gpu', devices=1, max_epochs=150,
            logger=logger, callbacks=callbacks,
            enable_progress_bar=self.test)
        
        # init and train
        model = TestNet(
            batch_size=b,
            continuous=c,
            drop_rate=d,
            learning_rate=r,
            weight_decay=w,
            val_batch_size=len(mnist_val))
        
        trainer.fit(model, train_loader, val_loader)
        return 0.5 # throw away value because we are not optimizing

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    outpath = os.path.join("data", "out")
    inpath = os.path.join("data", "in")
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action='store_true')

    search_space = {
        'batch_size': [16, 64],
        'drop_rate': [0.3, 0.5],
        'learning_rate': [0.001, 0.01], # max lr (@ maxepoch/10)
        'weight_decay': [0.5, 0.8],
        'continuous': [True, False]
    } 

    test = parser.parse_args().test
    objective = Objective(inpath, outpath, test)
    study0 = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space))
    study0.optimize(objective, n_trials=2 if test else 32)
    
    if not test:
        # Two study objects are necessary b/c gridsampler automatically
        # stops when possibilities are exhausted.
        study1 = optuna.create_study(
        sampler=optuna.samplers.GridSampler(search_space))
        study1.optimize(objective, n_trials=32)

    # transfer files to S3
    session = boto3.session.Session(profile_name="RegStudyUser")
    bucket = session.resource("s3").Bucket("reg-study-bucket")
    
    df = study0.trials_dataframe()
    if not test:
        df = pd.concat([df, study0.trials_dataframe()])

    csv_path = os.path.join(outpath, "trial_info.csv")
    df.to_csv(csv_path)
    bucket.upload_file(csv_path, "trial_info.csv")

    for t in range(2 if test else 64):
        tn = f"trial{t:02}"
        tpath = os.path.join(outpath, tn)
        bucket.upload_file(
            os.path.join(tpath, "train_stats", "version_0", "metrics.csv"),
            tn+"metrics.csv")
        
        for entry in os.listdir(tpath):
            if entry.split(".")[-1] == "ckpt":
                bucket.upload_file(os.path.join(tpath, entry), tn+entry)