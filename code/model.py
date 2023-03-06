import pytorch_lightning as pl
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, drop_rate):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.drop_rate = drop_rate
        self.add_skip = in_features==out_features

        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=drop_rate))
         
    def forward(self, x):
        z = self.layer(x)
        return (z+x) if self.add_skip else z


class TestNet(pl.LightningModule):

    def __init__(self, batch_size, drop_rate, learning_rate, weight_decay,
                 continuous=True, epochs=150, val_batch_size=None):

        """An heightened network for examining regularization
        in the continuous and discrete output settings; i.e. regression vs.
        classification.
        ARGS:
            batch_size - The number of observations per minibatch.
            drop_rate - The probability a node is zeroed in drop out.
            learning_rate - The maximum value of the learning rate.
            weight_decay - Scaling parameter for the L2 penalty.
            continuous - Should the output be treated as continuous or discrete.
            epochs - Total number of epochs for which to train the model.
        """
        super().__init__()
        
        self.bs = batch_size
        self.dr = drop_rate
        self.lr = learning_rate
        self.wd = weight_decay

        self.c1 = continuous
        self.epochs = epochs
        self.vbs = val_batch_size if val_batch_size else torch.tensor(self.bs)
        self.acc_num, self.acc_den = torch.tensor(0), torch.tensor(0)

        self.model = nn.Sequential(nn.Flatten())
        self.model.append(nn.Linear(28*28, 128))
        self.model.append(nn.ReLU())
        layer_params = [(28*28, 60), (60, 60)]
        for i, o in layer_params:
            self.model.append(LinearLayer(i, o, drop_rate))

        if continuous:
            self.model.append(nn.Linear(60, 1))
            self.loss = nn.MSELoss()
        else:
            self.model.append(nn.Sequential(
                nn.Linear(60, 10),
                nn.LogSoftmax(1)))
            self.loss = nn.NLLLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).squeeze()
        return self.loss(pred, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if self.c1:
            pred = pred.round().int().squeeze()
        else:
            pred = pred.argmax(dim=1)

        self.acc_num = self.acc_num + pred.eq(y).float().sum().detach().item()
        self.acc_den = self.acc_den + self.vbs
        self.log("acc", self.acc_num/self.acc_den, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, val_step_outputs):
        self.acc_num = torch.tensor(0)
        self.acc_den = torch.tensor(0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd)

        peak_lr_epoch = self.epochs // 10
        sch0 = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=peak_lr_epoch)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.epochs - peak_lr_epoch)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([sch0, sch1])
            
        return [optimizer], [scheduler]