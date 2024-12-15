import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import argparse
import logging

from models.utils import StandardScaler, DataLoader
from models.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
from models.transformer import TransformerModel
from models.lstm import LSTMModel

# logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)


def train_model(model, model_kind):
    model = model.to(device)
    modelpt_path = model_kind + '.pt'
    criterion = nn.L1Loss()  # MAE loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    criterion = nn.L1Loss()

    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader'].get_iterator()
        losses = []
        for x, y in data_iter:
            x, y = prepare_x_y(x, y)
            optimizer.zero_grad()
            output = model(x, args.horizon)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)

            loss = masked_mae_loss(y_pred, y_true)
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
            # if isinstance(model, LSTMModel):
            #     optimizer.zero_grad()
            #     output = model(x, args.horizon)
            #     y_pred = scaler.inverse_transform(output)
            #     y_true = scaler.inverse_transform(y)

            #     loss = masked_mae_loss(y_pred, y_true)
            #     losses.append(loss.item())
            #     batches_seen += 1
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            #     optimizer.step()

            # elif isinstance(model, TransformerModel):
            #     batch_size, seq_len, num_nodes, input_dim = x.size()
            #     for node in range(num_nodes):
            #         optimizer.zero_grad()
            #         _x, _y = x[:, :, node, :], y[:, :, node, :]
            #         output = model(_x, args.horizon)
            #         y_pred = scaler.inverse_transform(output)
            #         y_true = scaler.inverse_transform(_y)

            #         loss = masked_mae_loss(y_pred, y_true)
            #         losses.append(loss.item())
            #         batches_seen += 1
            #         loss.backward()
            #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            #         optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        print(message)
        test_loss, _, pred = evaluate(model, 'test')
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
            print(pred.shape)
            np.save(f'test_pred_{args.dataset}_{model_kind}.npy', pred)
            print('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                print('Early stopping at epoch: %d' % epoch_num)
                break
    
    print('=' * 35 + 'Best model performance' + '=' * 35)
    # model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, pred = evaluate(model, 'test')
    print(pred.shape)
    np.save(f'test_pred_{args.dataset}_{model_kind}.npy', pred)


def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        maes, mapes, mses = [], [], []
        l_1, m_1, r_1 = [], [], []
        l_2, m_2, r_2 = [], [], []
        l_3, m_3, r_3 = [], [], []
        for x, y in data_iter:
            x, y = prepare_x_y(x, y)
            output = model(x, args.horizon)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)

            loss = masked_mae_loss(y_pred, y_true)
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            maes.append(masked_mae_loss(y_pred, y_true).item())
            mapes.append(masked_mape_loss(y_pred, y_true).item())
            mses.append(masked_mse_loss(y_pred, y_true).item())
            # Important for MegaCRN model to let T come first.
            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
            m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
            r_1.append(masked_mse_loss(y_pred[0:1], y_true[0:1]).item())
            l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
            m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
            r_2.append(masked_mse_loss(y_pred[1:2], y_true[1:2]).item())
            l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
            # if isinstance(model, LSTMModel):
            #     output = model(x, args.horizon)
            #     y_pred = scaler.inverse_transform(output)
            #     y_true = scaler.inverse_transform(y)

            #     loss = masked_mae_loss(y_pred, y_true)
            #     losses.append(loss.item())
            #     # Followed the DCRNN TensorFlow Implementation
            #     maes.append(masked_mae_loss(y_pred, y_true).item())
            #     mapes.append(masked_mape_loss(y_pred, y_true).item())
            #     mses.append(masked_mse_loss(y_pred, y_true).item())
            #     # Important for MegaCRN model to let T come first.
            #     y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            #     l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
            #     m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
            #     r_1.append(masked_mse_loss(y_pred[0:1], y_true[0:1]).item())
            #     l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
            #     m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
            #     r_2.append(masked_mse_loss(y_pred[1:2], y_true[1:2]).item())
            #     l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            #     m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            #     r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            #     ys_true.append(y_true)
            #     ys_pred.append(y_pred)
            # elif isinstance(model, TransformerModel):
            #     batch_size, seq_len, num_nodes, input_dim = x.size()
            #     _ys_true = []
            #     _ys_pred = []
            #     for node in range(num_nodes):
            #         _x, _y = x[:, :, node, :], y[:, :, node, :]
            #         output = model(_x, args.horizon)
            #         y_pred = scaler.inverse_transform(output)
            #         y_true = scaler.inverse_transform(_y)

            #         loss = masked_mae_loss(y_pred, y_true)

            #         losses.append(loss.item())
            #         # Followed the DCRNN TensorFlow Implementation
            #         maes.append(masked_mae_loss(y_pred, y_true).item())
            #         mapes.append(masked_mape_loss(y_pred, y_true).item())
            #         mses.append(masked_mse_loss(y_pred, y_true).item())
            #         # Important for MegaCRN model to let T come first.
            #         y_true, y_pred = y_true.permute(1, 0, 2), y_pred.permute(1, 0, 2)
            #         l_1.append(masked_mae_loss(y_pred[0:1], y_true[0:1]).item())
            #         m_1.append(masked_mape_loss(y_pred[0:1], y_true[0:1]).item())
            #         r_1.append(masked_mse_loss(y_pred[0:1], y_true[0:1]).item())
            #         l_2.append(masked_mae_loss(y_pred[1:2], y_true[1:2]).item())
            #         m_2.append(masked_mape_loss(y_pred[1:2], y_true[1:2]).item())
            #         r_2.append(masked_mse_loss(y_pred[1:2], y_true[1:2]).item())
            #         l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            #         m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            #         r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            #         _ys_true.append(y_true)
            #         _ys_pred.append(y_pred)
            #     y_true = torch.stack(_ys_true, dim=2)
            #     y_pred = torch.stack(_ys_pred, dim=2)
            #     ys_true.append(y_true)
            #     ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(mses))
        l_1, m_1, r_1 = np.mean(l_1), np.mean(m_1), np.sqrt(np.mean(r_1))
        l_2, m_2, r_2 = np.mean(l_2), np.mean(m_2), np.sqrt(np.mean(r_2))
        l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))

        y_truths = torch.concat(ys_true, dim=1).cpu().numpy()
        y_preds = torch.concat(ys_pred, dim=1).cpu().numpy()

        if mode == 'test':
            print('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_mape, mean_rmse))
            print('Horizon 5mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_1, m_1, r_1))
            print('Horizon 10mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_2, m_2, r_2))
            print('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3))
        return mean_loss, y_truths, y_preds


def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., :args.input_dim]
    y0 = y[..., :args.output_dim]
    x0 = torch.from_numpy(x0).float()
    y0 = torch.from_numpy(y0).float()
    return x0.to(device), y0.to(device)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=3, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
# parser.add_argument('--seed', type=int, default=100, help='random seed.')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'{args.dataset}', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

data['train_loader'] = DataLoader(data['x_train'], data['y_train'], args.batch_size, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], args.batch_size, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], args.batch_size, shuffle=False)

def main():
    print(args.dataset, 'training and testing started', time.ctime())
    print('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    print('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    print('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    model = LSTMModel(1, 256, 3, 1)
    train_model(model, 'lstm')
    print('=' * 35 + '=' * 35)
    model = TransformerModel(1, 256, 8, 3, 1)
    train_model(model, 'transformer')

    print(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()