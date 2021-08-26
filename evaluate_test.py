import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import utils
import model.net as net
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from dataloader import *
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

logger = logging.getLogger('DeepAR_test.Eval')
init_params = utils.Params('../init_params.json')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=init_params.dataset, help='Name of the dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def evaluate(model, loss_fn, test_loader, params, data_dir, sample=True, plot_graph=False):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        data_dir: Directory to save predictions
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    dfObj = pd.DataFrame(columns=['Date', 'history','ground_truth', 'prediction'])

    with torch.no_grad():
      #plot_batch = np.random.randint(len(test_loader)-1)
      plot_batch = np.random.randint(len(test_loader))
      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels, dates) in enumerate(tqdm(test_loader)):
          #print('\n i, id_batch, labels)', i, id_batch, labels.size())
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)
          #print('\nbatch_size: ',batch_size)

          for t in range(params.test_predict_start):
              # if z_t is missing, replace it by output mu from the last time step
              zero_index = (test_batch[t,:,0] == 0)
              if t > 0 and torch.sum(zero_index) > 0:
                  test_batch[t,zero_index,0] = mu[zero_index]

              mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
              input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
              input_sigma[:,t] = v_batch[:, 0] * sigma
          #print('sample:', sample)
          if sample:
              samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
          else:
              sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
              raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)


          for batch_number in range(batch_size):

              label_plot = labels[batch_number].data.cpu().numpy()
              predict_mu = sample_mu[batch_number].data.cpu().numpy()
              predict_sigma = sample_sigma[batch_number].data.cpu().numpy()

              #plot_mu = np.concatenate((input_mu[batch_number].data.cpu().numpy(), predict_mu), axis=1)
              #plot_sigma = np.concatenate((input_sigma[batch_number].data.cpu().numpy(), predict_sigma), axis=1)
              #plot_metrics = {_k: _v[batch_number] for _k, _v in sample_metrics.items()}
              date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dates[batch_number][0].data.cpu().numpy()))
              #print('Date: ', date)
              #print('Prediction: ', type(predict_mu) , type(predict_mu[0]), predict_mu)
              #print('Label', type(label_plot) ,label_plot[91:])

              dfObj = dfObj.append({'Date': date, 'history': label_plot[:98], 'ground_truth': label_plot[91:], 'prediction':predict_mu }, ignore_index=True)

      dfObj.to_csv(os.path.join(data_dir, 'DeepAR_prediction_test.csv'))
      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name) 
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    utils.set_logger(os.path.join(model_dir, 'eval.log'))

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    # Create the input data pipeline
    logger.info('Loading the datasets...')

    test_set = Test_Final_Dataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('- done.')

    print('model: ', model)
    loss_fn = net.loss_fn

    logger.info('Starting evaluation')

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file+'_'+init_params.dataset+ '.pth.tar'), model)


    test_metrics = evaluate(model, loss_fn, test_loader, params, data_dir, params.sampling)
    #save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    #utils.save_dict_to_json(test_metrics, save_path)
