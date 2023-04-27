import logging
import logging.config
import yaml
import pandas as pd
import re
import math
import torch
import pyproj
import random
from tqdm import tqdm

logging.config.dictConfig(yaml.safe_load(open('./logconfig.yml', 'r')))
logger = logging.getLogger('logger')


def polyline_to_list(ori_str):
    str = ori_str.strip().replace('[', '').replace(']', '')
    traj = str.split(',')
    res = []
    if len(traj) % 2 != 0:
        return res
    i = 0
    while i < len(traj):
        point = []
        point.append(float(traj[i]))
        point.append(float(traj[i + 1]))
        res.append(point)
        i += 2
    return res


class DataProcessor():
    def __init__(self, bounding, crs_transformer, cell_len: float = 100):
        self.bounding = bounding
        self.crs_transformer = crs_transformer
        self.cell_len = cell_len
        min_x, min_y = self.crs_transformer.transform(self.bounding[1], self.bounding[0])
        max_x, max_y = self.crs_transformer.transform(self.bounding[3], self.bounding[2])
        row_nums, col_nums = math.ceil((max_y - min_y) / self.cell_len), math.ceil((max_x - min_x) / self.cell_len)
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.row_nums = row_nums
        self.col_nums = col_nums
        self.cell_nums = col_nums * row_nums

    def data_cleaning(self, raw_data_path: str, dst_data_path: str,
                      min_seq_len: int = 8, max_nums: int = None):
        min_x, min_y = self.crs_transformer.transform(self.bounding[1], self.bounding[0])
        max_x, max_y = self.crs_transformer.transform(self.bounding[3], self.bounding[2])
        row_nums, col_nums = math.ceil((max_y - min_y) / self.cell_len), math.ceil((max_x - min_x) / self.cell_len)
        logger.info('row nums: {}, col nums: {}, total cells: {}'.format(row_nums, col_nums, row_nums * col_nums))

        raw_frame = pd.read_csv(raw_data_path)
        trajectories = []
        cell_trajectories = []
        seq_nums = 0
        for i in tqdm(range(len(raw_frame))):
            # coords_list = polyline_to_list(raw_frame['POLYLINE'][i])
            coords_list = self.wkt_to_list(raw_frame['wkt'][i])
            if len(coords_list) < min_seq_len:
                continue
            in_region = True
            for point in coords_list:
                if point[0] < self.bounding[0] or point[0] > self.bounding[2] or \
                        point[1] < self.bounding[1] or point[1] > self.bounding[3]:
                    in_region = False
                    break
            if in_region:
                cell_ids = []
                for gps_point in coords_list:
                    x, y = self.crs_transformer.transform(gps_point[1], gps_point[0])
                    row, col = (y - min_y) // self.cell_len, (x - min_x) // self.cell_len
                    cell_ids.append(int(row * col_nums + col))
                trajectories.append([seq_nums, coords_list])
                cell_trajectories.append([seq_nums, cell_ids])
                seq_nums += 1
            if max_nums is not None and seq_nums >= max_nums:
                break

        logger.info('total trajectory nums: {}'.format(seq_nums))
        pd.DataFrame(data=trajectories, columns=['index', 'trajectory']).to_csv(dst_data_path, index=False)
        pd.DataFrame(data=cell_trajectories, columns=['index', 'cell_ids']).to_csv(dst_data_path.replace('.csv', '_cell.csv'), index=False)
        cfg = open(dst_data_path.replace('.csv', '_cfg.txt'), 'w')
        cfg.write('gps bounding: {}\nrow nums: {}, col nums: {}, total cells: {}\n'
                  .format(self.bounding, row_nums, col_nums, row_nums * col_nums))

    def id_to_loc(self, cid):
        """
        把栅格 ID 转换为行列
        :param cid:
        :return: (r, c)
        """
        return cid // self.col_nums, cid % self.col_nums

    def cal_cell_dist(self, src_cell, trg_cell):
        """
        计算栅格距离，以栅格变长为单位
        :param src_cell:
        :param trg_cell:
        :return:
        """
        r1, c1 = self.id_to_loc(src_cell)
        r2, c2 = self.id_to_loc(trg_cell)
        return int(math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))
    
    @staticmethod
    def wkt_to_list(line_str: str):
        spoints = line_str.replace('LineString(', '').replace(')', '').split(',')
        res_points = list()
        for pair in spoints:
            point = pair.split(' ')
            res_points.append([float(point[0]), float(point[1])])
        return res_points

    def cell_to_wkt_line(self, cell_id):
        r, c = cell_id // self.col_nums, cell_id % self.col_nums
        s_x, s_y = self.min_x + self.cell_len * c, self.min_y + self.cell_len * r
        t_x, t_y = s_x + self.cell_len, s_y + self.cell_len
        s_lat, s_lon = self.crs_transformer.transform(s_x, s_y, direction=pyproj.enums.TransformDirection.INVERSE)
        t_lat, t_lon = self.crs_transformer.transform(t_x, t_y, direction=pyproj.enums.TransformDirection.INVERSE)
        return 'LineString({} {},{} {},{} {},{} {},{} {})'.format(s_lon, s_lat, s_lon, t_lat, t_lon, t_lat,
                                                               t_lon, s_lat, s_lon, s_lat)

    def cell_to_wkt_polygon(self, cell_id):
        r, c = self.id_to_loc(cell_id)
        s_x, s_y = self.min_x + self.cell_len * c, self.min_y + self.cell_len * r
        t_x, t_y = s_x + self.cell_len, s_y + self.cell_len
        s_lat, s_lon = self.crs_transformer.transform(s_x, s_y, direction=pyproj.enums.TransformDirection.INVERSE)
        t_lat, t_lon = self.crs_transformer.transform(t_x, t_y, direction=pyproj.enums.TransformDirection.INVERSE)
        return 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(s_lon, s_lat, s_lon, t_lat, t_lon, t_lat,
                                                                  t_lon, s_lat, s_lon, s_lat)

    def view_trajectory(self, data_path: str, idx: int):
        gps_frame = pd.read_csv(data_path)
        cell_frame = pd.read_csv(data_path.replace('.csv', '_cell.csv'))
        assert idx < len(gps_frame)
        gps_tra = polyline_to_list(gps_frame['full_points'][idx])
        cell_tra = [int(k) for k in cell_frame['full_points'][idx].strip().replace('[', '').replace(']', '').split(',')]
        print(gps_tra)
        print(cell_tra)
        cell_res = []
        for c in cell_tra:
            cell_res.append(self.cell_to_wkt_line(c))

        pd.DataFrame(data=gps_tra, columns=['lon', 'lat']).to_csv('view_gps.csv', index=True)
        pd.DataFrame(data=cell_res, columns=['wkt']).to_csv('view_cells.csv', index=True)

    def output_trajectory(self, cell_ids: list, sample_idx: list, output_path: str):
        assert len(cell_ids) == len(sample_idx)
        res = list()
        for i, cid in enumerate(cell_ids):
            mask = 0 if sample_idx[i] != -1 else 1
            res.append([self.cell_to_wkt_polygon(cid), mask])
        pd.DataFrame(data=res, columns=['wkt', 'masked']).to_csv(output_path)

    def cal_mean_absolute_error(self, x_ids: list, y_ids: list, sample_idx: list):
        assert len(x_ids) == len(y_ids) == len(sample_idx)
        n = len(sample_idx)
        sum_error, rcv_cnt = 0, 0
        for i in range(n):
            if sample_idx[i] == -1:
                x_r, x_c = self.id_to_loc(x_ids[i])
                y_r, y_c = self.id_to_loc(y_ids[i])
                dist = math.sqrt((x_r - y_r) ** 2 + (x_c - y_c) ** 2) * self.cell_len
                sum_error += dist
                rcv_cnt += 1
        return sum_error / rcv_cnt, rcv_cnt


class TrajectoryDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        pass


def str_to_cell_list(cell_str: str):
        return [int(c) for c in cell_str.replace('[', '').replace(']', '').strip().split(',')]


def get_batches(data, batch_size, cell_nums, sample_ratio=0.5):
    """
    批量获取训练集数据，实现采样
    :param data:
    :param batch_size:
    :param sample_ratio:
    :return:
    """
    data_size = len(data) - batch_size

    for i in range(0, data_size, batch_size):
        min_len = math.inf
        ground_truth = []
        for j in range(0, batch_size):  # 获取该批次中序列的最小长度
            min_len = min(min_len, len(data[i + j]))
        for j in range(0, batch_size):
            ground_truth.append(data[i + j][0: min_len])
        sample_num = int(min_len * sample_ratio)
        sample_idx = random.sample(range(1, min_len - 1), sample_num)

        # mask = torch.ones(size=(batch_size, min_len), dtype=torch.long)
        # mask[:, [0] + sample_idx + [-1]] = 0
        # y = torch.tensor(ground_truth, dtype=torch.long)
        # x = torch.masked_fill(x, mask, -1)
        # yield x, y

        sample = [0]
        for u in sample_idx:
            sample.append(u)
        sample.append(min_len - 1)
        sample.sort()

        sample_ids = [-1 for i in range(min_len)]
        for j in sample:
            sample_ids[j] = j
        x = [[ground_truth[j][k] for k in sample] for j in range(0, batch_size)]
        # one_hot_prob = torch.zeros(batch_size, min_len, cell_nums)
        # for i in range(batch_size):
        #     for j in range(min_len):
        #         one_hot_prob[i, j, ground_truth[i][j]] = 1
        yield torch.tensor(x, dtype=torch.long), torch.tensor(ground_truth, dtype=torch.long), sample_ids


if __name__ == '__main__':
    # raw_data_path = '/home/lyd/Dataset/PortoTaxi/train.csv'
    # dst_data_path = './data/trajectories.csv'
    transformer = pyproj.Transformer.from_crs(4326, 32629)
    processor = DataProcessor(bounding=(-8.68500, 41.12, -8.59766, 41.198378), crs_transformer=transformer)
    print(processor.cell_nums)
    # processor.data_cleaning(raw_data_path, dst_data_path)

    # raw_data_path = '/home/lyd/Projects/Generator/generated_trajectories/TrajZZ_2.csv'
    # dst_data_path = './data/zz_trajectories_2.csv'
    transformer = pyproj.Transformer.from_crs(4326, 32649)
    processor = DataProcessor(bounding=(113.6079347, 34.7330714, 113.8507614, 34.8864656), crs_transformer=transformer, cell_len=200)
    # print(processor.row_nums)
    # print(processor.col_nums)
    print(processor.cell_nums)

    # print('{}, {}, {}'.format(processor.row_nums, processor.col_nums, processor.cell_nums))
    # processor.data_cleaning(raw_data_path, dst_data_path)


