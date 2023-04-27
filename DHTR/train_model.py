import logging
import logging.config
import yaml
import torch
from torch import nn, optim
import math
import matplotlib.pyplot as plt

from DHTR import DHTR, HybridLoss
from data_processing import get_batches

logging.config.dictConfig(yaml.safe_load(open('./logconfig.yml', 'r')))
logger = logging.getLogger('logger')


def train(model: DHTR, n_epochs: int, train_data, eval_data, device, batch_size):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = HybridLoss()
    epoch_loss = list()
    min_loss = math.inf
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        batch_nums = 0
        group_loss = 0
        logger.info('********** epoch: {} **********'.format(epoch))
        for x, y, sample_idx in get_batches(train_data, cell_nums=model.cell_nums, batch_size=batch_size):
            optimizer.zero_grad()
            res_prob, res_cells = model(x, y, sample_idx)
            recover_idx = [j for j, k in enumerate(sample_idx) if k == -1]
            loss = criterion(res_prob, y, recover_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            train_loss += loss.item()
            group_loss += loss.item()
            if batch_nums >= 100 and batch_nums % 100 == 0:
                logger.info('epoch-{}, {} batch loss: {:.6f}'.format(epoch, batch_nums, group_loss / 100))
                group_loss = 0
            batch_nums += 1
        train_loss = train_loss / batch_nums

        model.eval()
        eval_loss = 0
        batch_nums = 0
        for x, y, sample_idx in get_batches(eval_data, cell_nums=model.cell_nums, batch_size=batch_size):
            res_prob, res_cells = model(x, y, sample_idx)
            recover_idx = [j for j, k in enumerate(sample_idx) if k == -1]
            loss = criterion(res_prob, y, recover_idx)
            eval_loss += loss.item()
            batch_nums += 1
        eval_loss = eval_loss / batch_nums
        epoch_loss.append([train_loss, eval_loss])
        if eval_loss < min_loss:
            min_loss = eval_loss
            torch.save(model, './saved_models/DHTR0401/DHTR_{}_loss_{:.2f}.pth'.format(epoch, min_loss))
        logger.info('epoch {}/{}, train loss = {:.6f}, eval loss = {:.6f}'
              .format(epoch, n_epochs, epoch_loss[epoch][0], epoch_loss[epoch][1]))




