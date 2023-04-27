import logging
import logging.config
import yaml
import torch
import random
import pyproj
import pandas as pd
from tqdm import tqdm

from data_processing import DataProcessor, get_batches, str_to_cell_list

logging.config.dictConfig(yaml.safe_load(open('./logconfig.yml', 'r')))
logger = logging.getLogger('logger')

def evaluate(model, data_processor, test_dataset: list, visual_nums: int = 10):
    visual_ids = random.sample(range(len(test_dataset)), visual_nums)
    with torch.no_grad():
        model.eval()
        sum_error, sum_weight, idx = 0, 0, 0
        for x, y, sample_ids in get_batches(data=test_dataset, batch_size=1, cell_nums=data_processor.cell_nums, sample_ratio=0.5):
            res_prob, res_cells = model(x, y, sample_ids, teacher_forcing_ratio=0)
            px = res_cells[0].long().cpu().tolist()
            py = y[0].long().cpu().tolist()
            mae, rcv_cnt = processor.cal_mean_absolute_error(px, py, sample_ids)
            
            if idx in visual_ids:
                path = './test_results/test_{}.csv'.format(idx)
                data_processor.output_trajectory(list(res_cells[0].numpy()), sample_ids, path)
            
            sum_error += rcv_cnt * mae
            sum_weight += rcv_cnt
            idx += 1
            if idx % 1000 == 0:
                logger.info('testing %d/%d, avg MAE = %.2f' % (idx, len(test_dataset), sum_error / sum_weight))
                # print('testing %d/%d, avg MAE = %.2f' % (idx, len(test_dataset), sum_error / sum_weight)



def load_traj(cell_ids, sample_ratio):
    seq_len = len(cell_ids)
    sample_num = int(seq_len * sample_ratio)
    sample = random.sample(range(1, seq_len - 1), sample_num)
    sample.insert(0, 0)
    sample.append(seq_len - 1)
    sample.sort()

    sample_idx = [-1 for i in range(seq_len)]
    for j in sample:
        sample_idx[j] = j
    
    x = torch.tensor([cell_ids[k] for k in sample]).unsqueeze(0)
    y = torch.tensor(cell_ids).unsqueeze(0)
    # y = torch.zeros(size=(1, seq_len))
    return x, y, sample_idx


if __name__ == '__main__':
    transformer = pyproj.Transformer.from_crs(4326, 32629)
    processor = DataProcessor(bounding=(-8.68500, 41.12, -8.59766, 41.198378), crs_transformer=transformer)
    
    # transformer = pyproj.Transformer.from_crs(4326, 32649)
    # processor = DataProcessor(bounding=(113.6079347, 34.7330714, 113.8507614, 34.8864656), crs_transformer=transformer, cell_len=200)
    
    model = torch.load('./saved_models/DHTR0312/DHTR_29_loss_0.56.pth')
    # model = torch.load('./saved_models/DHTR0401/DHTR_29_loss_0.69.pth')

    # model.eval()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_frame = pd.read_csv('./data/trajectories_cell.csv')
    data = [str_to_cell_list(data_frame['cell_ids'][i]) for i in range(len(data_frame))]
    
    evaluate(model, processor, data[int(len(data) * 0.8): ])

    # for x, y, sample_idx in get_batches(data[int(len(data) * 0.8): ], cell_nums=model.cell_nums, batch_size=1):
    #     res_prob, res_cell = model(x, y, sample_idx, teacher_forcing_ratio=0)
    #     px = res_cell[0].long().cpu().tolist()
    #     py = y[0].long().cpu().tolist()
    #     mae, rcv_cnt = processor.cal_mean_absolute_error(px, py, sample_idx)
    #     print('seq_len=%d, MAE=%.2f, rcv_cnt=%d' % (y.shape[1], mae, rcv_cnt))
    #     processor.output_trajectory(list(res_cell[0].cpu().numpy()), sample_idx, './output/res.csv')
    #     processor.output_trajectory(list(y[0].cpu().numpy()), sample_idx, './output/traj.csv')
    #     break

    # idx = int(len(data) * 0.8) + 8
    # x, y, sample_idx = load_traj(data[idx], 0.5)
    # x = x.to(device)
    # y = y.to(device)
    # y0 = torch.zeros(size=(1, len(sample_idx))).to(device)
    # res_prob, res_cell = model(x, y, sample_idx, teacher_forcing_ratio=0)

    # processor.output_trajectory(list(res_cell[0].cpu().numpy()), sample_idx, './output/res.csv')
    # processor.output_trajectory(list(y[0].cpu().numpy()), sample_idx, './output/traj.csv')

    # print(x)
    # print(y)
    # print(sample_idx)
    # evaluate(model, processor, data[int(len(data) * 0.8): int(len(data) * 0.8) + 20])

    