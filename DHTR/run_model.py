import logging
import logging.config
import yaml
import torch
import pyproj
import pandas as pd


from data_processing import DataProcessor
from DHTR import DHTR
from train_model import train

logging.config.dictConfig(yaml.safe_load(open('./logconfig.yml', 'r')))
logger = logging.getLogger('logger')


if __name__ == '__main__':
    transformer = pyproj.Transformer.from_crs(4326, 32649)
    processor = DataProcessor(bounding=(113.6079347, 34.7330714, 113.8507614, 34.8864656), crs_transformer=transformer, cell_len=200)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DHTR(embedding_dim=512, lstm_layers=2, dropout=0.5, attn_dim=256, max_dist=1024,
                 device=device, row_nums=processor.row_nums, col_nums=processor.col_nums)
    # model = torch.load('./saved_models/DHTR0325/DHTR_29_loss_0.55.pth')
    
    def str_to_cell_list(cell_str: str):
        return [int(c) for c in cell_str.replace('[', '').replace(']', '').strip().split(',')]
    
    data_frame = pd.read_csv('./data/zz_trajectories_2_cell.csv')
    data = [str_to_cell_list(data_frame['cell_ids'][i]) for i in range(len(data_frame))]
    train_data, eval_data = data[: int(len(data) * 0.7)], data[int(len(data) * 0.7): int(len(data) * 0.8)]
    
    train(model=model, n_epochs=30, train_data=train_data, eval_data=eval_data, device=device, batch_size=64)





