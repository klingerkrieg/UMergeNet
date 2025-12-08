#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from util import *
import pandas as pd 
from datetime import timedelta
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from util import count_trainable_parameters, measure_inference_speed

#pip install XlsxWriter
#jupyter nbconvert --to script Trainer.ipynb
#Version 1.2


# In[ ]:


class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if self.mode == 'min':
            self.sign = 1
        else:  # 'max'
            self.sign = -1

    def step(self, score):
        score = self.sign * score  # transform max into min if necessary

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# In[ ]:


import torch
import numpy as np
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
# We are not using DiceScore for multiclass here
# because they don't have ignore_index support yet.
#from torchmetrics.segmentation import DiceScore, MeanIoU


def evaluate_model(model, data_loader, num_classes=2, print_stats=False, criterion=None, device='cuda', ignore_index=None):
    model.eval()
    #dice_metric      = DiceScore(num_classes=num_classes, average="macro", input_format='index', aggregation_level='global').to(device)
    #miou_metric      = MeanIoU(num_classes=num_classes, input_format='index').to(device)

    # these metrics were computed per class and averaged = average="macro"
    miou_metric      = MulticlassJaccardIndex(num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)
    prec_metric      = MulticlassPrecision(num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)
    f1_metric        = MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=ignore_index).to(device)

    val_loss    = 0.0
    mIoU        = 0.0
    precision   = 0.0
    recall      = 0.0
    f1          = 0.0
    q           = 0.0

    dataset_size = len(data_loader.dataset)

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks  = masks.to(device)
            if num_classes > 2:
                masks = masks.squeeze(1)

            outputs = model(images)

            # Loss 
            if criterion is not None:
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

            # sigmoid for binary, softmax for multi-class
            if num_classes == 2:
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).long()
            else:
                preds = torch.argmax(outputs, dim=1)

            masks = masks.long()

            # TorchMetrics
            batch_mIoU      = miou_metric(preds, masks).item()
            batch_precision = prec_metric(preds, masks).item()
            batch_recall    = recall_metric(preds, masks).item()
            batch_f1        = f1_metric(preds, masks).item()

            batch_size = images.size(0)
            mIoU      += batch_mIoU      * batch_size
            precision += batch_precision * batch_size
            recall    += batch_recall    * batch_size
            f1        += batch_f1        * batch_size
            # Q = Dice * mIoU
            q         += (batch_f1 * batch_mIoU) * batch_size

    # Averages
    avg_loss        = val_loss    / dataset_size if criterion else 0.0
    avg_f1          = f1          / dataset_size
    avg_mIoU        = mIoU        / dataset_size
    avg_precision   = precision   / dataset_size
    avg_recall      = recall      / dataset_size
    avg_q           = q           / dataset_size

    if print_stats:
        print(
            f"Loss: {avg_loss:.4f} "
            f"F1: {avg_f1:.4f} mIoU: {avg_mIoU:.4f} "
            f"Prec: {avg_precision:.4f} "
            f"Recall: {avg_recall:.4f} Q: {avg_q:.4f}"
        )

    return {'loss':avg_loss, 'f1':avg_f1, 'miou':avg_mIoU, 'precision':avg_precision, 'recall':avg_recall, 'q':avg_q}


# In[ ]:


from enum import Enum

class Losses(Enum):
    CrossEntropyLoss    = 0
    BCEWithLogitsLoss   = 1
    BCEDiceLoss         = 2

#BCE + Dice Loss (for binary segmentation)
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # ECB (with logits)
        bce_loss = self.bce(inputs, targets)

        # Sigmoid to convert logits → probabilities
        probs = torch.sigmoid(inputs)

        # Flatten for Dice calculation
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # Thoughtful combination
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return loss
    


# In[ ]:


from types import FunctionType


class Trainer:

    model         = None
    criterion     = None
    optimizer     = None
    scheduler     = None
    learning_rate = None
    ignore_index  = None

    def __init__(self, 
                 model_filename=None, 
                 model_dir=None, 
                 info={}, 
                 save_xlsx=False, 
                 load_best=True, 
                 device=None, 
                 rewrite_model=False, 
                 num_classes = 2,
                 loss_function=Losses.BCEDiceLoss,
                 ignore_index=None):

        if save_xlsx:
            if model_filename is None:
                raise Exception("model_filename is mandatory when with save_xlsx == True")
        
        self.save_xlsx     = save_xlsx
        self.load_best     = load_best
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.loss_function = loss_function

        # saves the model name and directory
        self.model_filename = model_filename
        if model_dir is None:
            model_dir = model_filename
        self.model_dir = model_dir

        # if at least the model name is passed
        if self.model_filename is not None:
            self.model_file_dir = self.model_dir + "/" + self.model_filename
            self.hist_name = self.model_file_dir.replace('.pth', '.xlsx')
            self.best_path           = self.model_file_dir.replace('.pth', '-best.pth')
            self.last_path           = self.model_file_dir.replace('.pth', '-last.pth')
        else:
            self.model_file_dir = None

        if rewrite_model and self.model_file_dir is not None:
            if os.path.exists(self.hist_name):
                os.remove(self.hist_name)
            if os.path.exists(self.model_file_dir):
                os.remove(self.model_file_dir)
            if os.path.exists(self.best_path):
                os.remove(self.best_path)
            if os.path.exists(self.last_path):
                os.remove(self.last_path)

        # extra information to be saved in xlsx
        self.info = info
        # index of the sample image that will be used
        # to save output during training
        self.sample_img_fixed_index = 0
        # Makes some initializations
        self.create_criterion()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Device:",self.device)
    

    def create_criterion(self):
        # If the loss is present in the Losses enum
        if isinstance(self.loss_function, Losses):    
            self.info['loss_function'] = self.loss_function

            if self.loss_function == Losses.CrossEntropyLoss:
                self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            elif self.loss_function == Losses.BCEWithLogitsLoss:
                self.criterion = nn.BCEWithLogitsLoss()
            elif self.loss_function == Losses.BCEDiceLoss:
                self.criterion = BCEDiceLoss(bce_weight=0.5)

        # otherwise you cand pass a custom function
        elif isinstance(self.loss_function, FunctionType):
            self.info['loss_function'] = self.loss_function.__name__
            self.criterion = self.loss_function

        elif isinstance(self.loss_function, object):
            self.info['loss_function'] = self.loss_function.__class__.__name__
            self.criterion             = self.loss_function
        
            
        
        
    
    def create_scheduler(self, patience=10, factor=0.5, mode='max'):
        self.info['scheduler'] = "ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, verbose=True)"
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                      mode=mode, 
                                      patience=patience, 
                                      factor=factor)
        
        

    def create_optimizer(self):
        self.info['optimizer'] = f"optim.Adam(self.model.parameters(), lr={self.learning_rate})"
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    
    
    def train_loop(self, images, masks, epoch):
        outputs     = self.model(images)

        if self.num_classes == 2:
            outputs   = outputs.squeeze(1)
            masks_s   = masks.squeeze(1).float()
        else:
            outputs   = self.model(images)
            masks_s   = masks.long().squeeze(1)

        loss    = self.criterion(outputs, masks_s)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss = loss.item() * images.size(0)

        return train_loss
    
    


    def update_history(self, history, train_loss=None, loss=None, f1=None, miou=None,
                            precision=None, recall=None, q=None, 
                            elapsed_time=None, images_per_sec=None, started=None):
        if train_loss is not None:
            history["train_loss"].append(train_loss)
        if loss is not None:
            history["loss"].append(loss)
        if f1 is not None:
            history["f1"].append(f1)
        if miou is not None:
            history["miou"].append(miou)
        if precision is not None:
            history["precision"].append(precision)
        if recall is not None:
            history["recall"].append(recall)
        if q is not None:
            history["q"].append(q)
        if elapsed_time is not None:
            history["elapsed_time"].append(elapsed_time)
        if images_per_sec is not None:
            history["images_per_sec"].append(images_per_sec)
        if started is not None:
            history["started"].append(started)


    
    def print_last_history_stats(self):
            def format_line(title, epoch_idx):
                epoch = epoch_idx + 1
                values = {k: self.val_history[k][epoch_idx] for k in self.val_history}
                lr = self.val_history.get("lr", [None] * len(self.val_history["loss"]))[epoch_idx]
                gpu_fps = values.get("GPU_FPS", None)
                cpu_fps = values.get("CPU_FPS", None)
                line = (
                    f"{title}:\n"
                    f" Epoch [{epoch}]"
                    f" - Loss: {values.get('train_loss', float('nan')):.4f}"
                    f" Val Loss: {values.get('loss', float('nan')):.4f}"
                    f" F1-score: {values.get('f1', float('nan')):.4f}"
                    f" mIoU: {values.get('miou', float('nan')):.4f}"
                    f" Precision: {values.get('precision', float('nan')):.4f}"
                    f" Recall: {values.get('recall', float('nan')):.4f}"
                    f" Q: {values.get('q', float('nan')):.4f}"
                    f" Tempo total: {values.get('elapsed_time', 'nan')}"
                )
                if lr is not None:
                    line += f" LR:{lr:.6f}"
                if gpu_fps is not None:
                    line += f" GPU_FPS: {gpu_fps:.2f}"
                if cpu_fps is not None:
                    line += f" CPU_FPS: {cpu_fps:.2f}"
                return line

            # best time (highest F1)
            best_epoch = int(max(range(len(self.val_history["f1"])), key=lambda i: self.val_history["f1"][i]))
            print(format_line("Best model", best_epoch))

            # last season
            last_epoch = len(self.val_history["f1"]) - 1
            print(format_line("Latest model", last_epoch))


    def do_save_xlsx(self):

        avg_speed = sum(self.val_history['images_per_sec']) / len(self.val_history['images_per_sec'])
        self.info['training_speed_img_per_sec'] = round(avg_speed, 2)

        df_val_history = pd.DataFrame(self.val_history)
        df_val_history.insert(0, 'epoch', range(1, len(df_val_history)+1))
        df_val_history['epoch'] = df_val_history['epoch'].astype(str)


        df_test_history = pd.DataFrame(self.test_history)
        df_test_history.insert(0, 'epoch', range(1, len(df_test_history)+1))
        df_test_history['epoch'] = df_test_history['epoch'].astype(str)


        df_info = pd.DataFrame(self.info, index=[0])
        with pd.ExcelWriter(self.hist_name, engine='xlsxwriter') as writer:
            df_val_history.to_excel(writer, sheet_name='val_history', index=False, float_format="%.4f")
            df_test_history.to_excel(writer, sheet_name='test_history', index=False, float_format="%.4f")
            df_info.to_excel(writer, sheet_name='model_info', index=False, float_format="%.4f")

            workbook  = writer.book
            worksheet = writer.sheets['val_history']

            chart = workbook.add_chart({'type': 'line'})

            # The 'epoch' column is now in column 0
            # Assuming 'val_f1' is in column 5 and 'val_IoU' in 6 (or adjust this dynamically)
            col_f1 = df_val_history.columns.get_loc('f1')
            col_iou  = df_val_history.columns.get_loc('miou')

            chart.add_series({
                'name':       'f1',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],  # column 0 = epoch
                'values':     ['val_history', 1, col_f1, len(df_val_history), col_f1],
            })
            chart.add_series({
                'name':       'mIoU',
                'categories': ['val_history', 1, 0, len(df_val_history), 0],
                'values':     ['val_history', 1, col_iou, len(df_val_history), col_iou],
            })

            chart.set_title({'name': 'Training'})

            chart.set_x_axis({
                'name': 'Epoch',
                'interval_unit': 10,
                'num_font': {'rotation': -45},
            })
            chart.set_y_axis({'name': 'Value'})

            worksheet.insert_chart('K2', chart)

    
    
    def load_xlsx_history(self):
        # If xlsx didn't exists, just return 0 and 0 and the Trainer will create a new one
        if os.path.exists(self.hist_name) == False:
            return 0, 0
        
        # Read all sheets in the file
        xls = pd.read_excel(self.hist_name, sheet_name=None)

        # Retrieves the history DataFrame and converts it to a dictionary list
        df_val_history   = xls['val_history']
        last_epoch   = int(df_val_history['epoch'].iloc[-1])
        self.val_history = df_val_history.drop(columns=['epoch']).to_dict(orient='list')


        df_test_history   = xls['test_history']
        self.test_history = df_test_history.drop(columns=['epoch']).to_dict(orient='list')

        # Accumulated time
        elapsed_str      = df_val_history['elapsed_time'].iloc[-1]
        h, m, s          = map(int, elapsed_str.split(':'))
        accumulated_time = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
        start_time       = time.time() - accumulated_time  # Adjusts to maintain accumulated count

        # Retrieves model information DataFrame and converts to dictionary
        df_info = xls['model_info']
        self.info = df_info.iloc[0].to_dict()
        return last_epoch, start_time

    def load_model(self, model_file_dir, model=None, load_xlsx=True, load_scheduler=False):
        #if the model is passed
        if model is not None:
            #self.model receives the new model
            self.model = model
        #if the model to be loaded has not been passed
        if self.model is None:
            raise Exception("You need to pass the model object in the 'model' parameter")
        
        if self.optimizer is None:
            self.create_optimizer()
        if self.scheduler is None:
            self.create_scheduler()
        
        #loads the model from the .pth file
        checkpoint = torch.load(model_file_dir, weights_only=False)
        #retrieves the states of the file
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_score  = checkpoint['best_acc']
        epoch       = checkpoint['epoch'] + 1
        self.model.to(self.get_device())
        print(f"Loaded model: {model_file_dir}")
        if load_xlsx:
            start_epoch, start_time = self.load_xlsx_history()
            return best_score, epoch, start_epoch, start_time
        return best_score, epoch
    
    def save_model(self, path, epoch, best_score):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_acc': best_score
                }, path)
    
    def get_device(self):
        return self.device


    def get_best_test_stats(self):
        return self.best_test_eval
        

    def train(self, model, 
                    train_loader, 
                    val_loader,
                    test_loader, 
                    num_epochs=50, 
                    # saves the model every
                    save_every=None,
                    # prints the tempo every
                    print_every=None,
                    # continue training where you left off
                    continue_from_last=False,
                    # verbose==1 prints the training on the same line
                    verbose=3,
                    learning_rate=1e-4,
                    # scheduler patience=decreases IR after 10 epochs without improvement in acc
                    scheduler_patience=10,
                    # early_stop_patience=ends training after 20 epochs if acc improves
                    early_stop_patience=20,
                    measure_cpu_speed=True,
                    print_val_stats=False,
                    # run evaluate_model after loading the best or last version instead of print xlst stats
                    re_evaluate=False
                    ):

        torch.backends.cudnn.benchmark = True
        device = self.get_device()

        self.learning_rate  = learning_rate
        self.model          = model
        start_epoch         = 0
        best_score          = -1.0
        start_time          = time.time()
        started             = False
        batch_size          = train_loader.batch_size

        trainable_parameters = count_trainable_parameters(model)
        print("Trainable_parameters:", trainable_parameters)
        print("Loss function:", self.loss_function)
        self.info['dataset_name']         = train_loader.dataset.__module__
        self.info['dataset_batch_size']   = batch_size
        self.info['trainable_parameters'] = trainable_parameters
        images, labels = next(iter(train_loader))
        self.info['dataset_resolution']   = f"{images.shape[2]} x {images.shape[3]}"
        

        self.val_history = {
            "train_loss":     [],
            "loss":           [],
            "f1":             [],
            "miou":           [],
            "precision":      [],
            "recall":         [],
            "q":              [],
            "elapsed_time":   [],
            "images_per_sec": [],
            "started":        [],
        }
        self.test_history = {k: [] for k in self.val_history}
        

        #prints everything on the same line
        tqdm_disable = print_every!=None
        print_end    = '\r\n'
        if verbose == 1:
            print_end    = '\r'
            tqdm_disable = True


        
        #if the model name was passed
        if self.model_filename is not None:
            #create directories
            os.makedirs(self.model_dir, exist_ok=True)

            #First, it checks whether the final trained model already exists
            if os.path.exists(self.model_file_dir):

                if re_evaluate:
                    print("Reevaluate is on")
                    print("Trained model already exists. \nLoading latest version.")
                    self.load_model(self.model_file_dir)
                    print("Evaluating the latest version...")
                    last_eval = evaluate_model(self.model, test_loader, num_classes=self.num_classes, print_stats=True, device=device, ignore_index=self.ignore_index)

                    print("Loading best version.")
                    self.load_model(self.best_path)
                    print("Evaluating the best version...")
                    best_eval = evaluate_model(self.model, test_loader, num_classes=self.num_classes, print_stats=True, device=device, ignore_index=self.ignore_index)

                    # If last version has f1 > than best version f1
                    if last_eval['f1'] > best_eval['f1']:
                        print(f"Latest version has the best f1-score on [testset]: latest({last_eval['f1']:.4f}) best({best_eval['f1']:.4f})")
                        self.load_model(self.model_file_dir)
                        print("Latest version loaded.")
                        self.best_test_eval = last_eval
                    else:
                        print(f"Best version has the best f1-score on [testset]: best({best_eval['f1']:.4f}) latest({last_eval['f1']:.4f})")
                        print("Best version loaded.")
                        self.best_test_eval = best_eval
                else:
                    print("Reevaluate is off")
                    print("Trained model already exists. \nLoading latest version.")
                    self.load_model(self.model_file_dir)
                    print("Printing last history stats:")
                    self.print_last_history_stats()
                
                return model
            #if it does not exist and is a continuation of the training
            elif continue_from_last == True:
                #continues from -last
                if os.path.exists(self.last_path):
                    _, _, start_epoch, start_time = self.load_model(self.last_path)
                    print(f"Continuing from the saved model: {self.last_path}")
                    print(f"start_epoch: {start_epoch}, start_time: {start_time}")
                    if start_epoch >= num_epochs:
                        self.print_last_history_stats()
                        return self.model
            


        model.to(device)
        self.create_optimizer()
        self.create_scheduler(patience=scheduler_patience)
        early_stopper = EarlyStopping(patience=early_stop_patience, mode='max')
    
        

        ## Training
        epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):
            
            model.train()
            train_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", disable=tqdm_disable):
                images = images.to(device)
                masks  = masks.to(device)
                ## training loop
                train_loss += self.train_loop(images, masks, epoch)

            avg_train_loss = train_loss / len(train_loader.dataset)
            

            ## Validation
            avg_val_loss, avg_val_f1, avg_val_mIoU, avg_val_precision, avg_val_recall, avg_val_q = evaluate_model(self.model, val_loader, num_classes=self.num_classes, criterion=self.criterion, ignore_index=self.ignore_index).values()

            ## Test 
            last_test_eval = evaluate_model(self.model, test_loader, num_classes=self.num_classes, criterion=self.criterion, ignore_index=self.ignore_index)
            avg_test_loss, avg_test_f1, avg_test_mIoU, avg_test_precision, avg_test_recall, avg_test_q = last_test_eval.values()

            elapsed     = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            current_lr  = self.optimizer.param_groups[0]['lr']

            
            
            test_stats = (f"Epoch [{epoch+1}/{num_epochs}] [test_set]: " 
                    f"Loss: {avg_train_loss:.4f}  " 
                    f"F1: {avg_test_f1:.4f} mIoU: {avg_test_mIoU:.4f} " 
                    f"Prec: {avg_test_precision:.4f} " 
                    f"Recall: {avg_test_recall:.4f} Q: {avg_test_q:.4f} " 
                    f"Time: {elapsed_str} LR:{current_lr:.6f}")
            
            # The validation stats is used only during training
            # for papers we use the test_stats, so we will print only test_stats
            # but you can print val_stats and test_stats together with print_val_stats=True
            val_stats = (f" - [val_set]: Loss: {avg_val_loss:.4f} " 
                    f"F1: {avg_val_f1:.4f} mIoU: {avg_val_mIoU:.4f} " 
                    f"Prec: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} ")

            if print_val_stats:
                test_stats += val_stats

            if print_every is None:
                print(test_stats,end=print_end)
            else:
                if (epoch+1) % print_every == 0:
                    print(val_stats,end=print_end)
                    print(test_stats,end=print_end)
            

            images_per_sec = (len(train_loader) * batch_size) / elapsed
            
            ## Saves the evolution of the network
            self.update_history(
                self.val_history,
                train_loss=avg_train_loss,
                loss=avg_val_loss,
                f1=avg_val_f1,
                miou=avg_val_mIoU,
                precision=avg_val_precision,
                recall=avg_val_recall,
                q=avg_val_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )
            self.update_history(
                self.test_history,
                train_loss=avg_train_loss,
                loss=avg_test_loss,
                f1=avg_test_f1,
                miou=avg_test_mIoU,
                precision=avg_test_precision,
                recall=avg_test_recall,
                q=avg_test_q,
                elapsed_time=elapsed_str,
                images_per_sec=images_per_sec,
                started=('started' if not started else '')
            )
            

            started = True

            # The avg_val_f1 will be observed for the scheduler and early_stopper

            # reduces the learning rate if the score does not improve
            self.scheduler.step(avg_val_f1)

            # for training if you don't improve in X times
            early_stopper.step(avg_val_f1)
            if early_stopper.early_stop:
                print(f"Stopping at epoch {epoch+1} by early stopping.")
                break

            ## Save the best model so far
            if avg_val_f1 > best_score:
                best_score = avg_val_f1

                # if best_test_stats is empty
                if self.best_test_eval is None:
                    self.best_test_eval = last_test_eval
                # if last_test_eval is better than best_test_eval 
                elif self.best_test_eval['f1'] < last_test_eval['f1']: #f1 = 1
                    # override best_test_eval
                    self.best_test_eval = last_test_eval


                
                if self.model_file_dir is not None:
                    #save the model at the best time
                    self.save_model(self.best_path, epoch, best_score)
                    current_lr  = self.optimizer.param_groups[0]['lr']
                                        
                    best_test_stats = (f"Epoch [{epoch+1}/{num_epochs}] [test_set]:" 
                                f"Loss: {avg_train_loss:.4f} Val Loss: {avg_test_loss:.4f} " 
                                f"F1: {avg_test_f1:.4f} mIoU: {avg_test_mIoU:.4f} " 
                                f"Prec: {avg_test_precision:.4f} " 
                                f"Recall: {avg_test_recall:.4f} Q: {avg_test_q:.4f} " 
                                f"Time: {elapsed_str} LR:{current_lr:.6f}")
                    
                    best_val_stats = (f" - [val_set]: Loss: {avg_val_loss:.4f} " 
                                f"F1: {avg_val_f1:.4f} mIoU: {avg_val_mIoU:.4f} " 
                                f"Prec: {avg_val_precision:.4f} " 
                                f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} ")
                    

                    if print_val_stats:
                        best_test_stats += best_val_stats
                    if print_every is None and verbose > 1:
                        print("✔ Best model saved:", best_test_stats, end=print_end)
                    #save excel to the current moment
                    if self.save_xlsx:
                        self.do_save_xlsx()
            
                

            #Saves the network every
            if save_every is not None and (epoch + 1) % save_every == 0:
                last_model_file_dir = self.model_file_dir.replace('.pth','-last.pth')
                self.save_model(last_model_file_dir, epoch, best_score)
                self.do_save_xlsx()
                if verbose > 1:
                    print("Saved last as", last_model_file_dir, end=print_end)


                
        last_test_stats = (f"Epoch [{epoch+1}/{num_epochs}] [test_set]: " 
                    f"Loss: {avg_train_loss:.4f} Val Loss: {avg_test_loss:.4f} " 
                    f"F1: {avg_test_f1:.4f} mIoU: {avg_test_mIoU:.4f} " 
                    f"Prec: {avg_test_precision:.4f} " 
                    f"Recall: {avg_test_recall:.4f} Q: {avg_test_q:.4f} " 
                    f"Time: {elapsed_str} LR:{current_lr:.6f}")
        
        last_val_stats = (f" - [val_set]: Loss: {avg_val_loss:.4f} " 
                    f"F1: {avg_val_f1:.4f} mIoU: {avg_val_mIoU:.4f} " 
                    f"Prec: {avg_val_precision:.4f} " 
                    f"Recall: {avg_val_recall:.4f} Q: {avg_val_q:.4f} ")
        if print_val_stats:
            last_test_stats += last_val_stats


        #calculates the FPS of the model
        self.info['GPU_FPS'], self.info['GPU_time_per_image'], self.info['CPU_FPS'], self.info['CPU_time_per_image'] = measure_inference_speed(self.model, 
                                                                                                                                               val_loader, 
                                                                                                                                               measure_cpu_speed=measure_cpu_speed)
          
        print("")
        if best_test_stats:
            print("Best model:\r\n", best_test_stats)
        print("Latest model:\r\n", last_test_stats + '\r\n GPU_FPS:',self.info['GPU_FPS'], ' CPU_FPS:',self.info['CPU_FPS'])


        if self.model_file_dir is not None:
            self.save_model(self.model_file_dir, epoch, best_score)
            print("Saved as", self.model_file_dir)

        
        if self.save_xlsx:
            # Write the excel file with history
            self.do_save_xlsx()

        #beep win
        #os.system('powershell.exe -Command "[console]::beep(600,200); [console]::beep(600,200);"')
        #linux
        os.system('play -nq -t alsa synth 0.2 sine 600; play -nq -t alsa synth 0.2 sine 600')
        return model

if __name__ == '__main__':
    pass

