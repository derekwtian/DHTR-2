import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DHTR(nn.Module):
    def __init__(self, embedding_dim, lstm_layers, dropout, attn_dim, max_dist, device, row_nums, col_nums):
        super().__init__()
        self.device = device
        self.row_nums = row_nums
        self.col_nums = col_nums
        self.max_dist = max_dist
        self.cell_nums = row_nums * col_nums
        # Encoder 部分
        self.loc_embedding = nn.Embedding(self.cell_nums, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim // 2, num_layers=lstm_layers,
                               dropout=dropout, bidirectional=True, batch_first=True)

        # ST Attention 部分
        self.spatial_dist_embedding = nn.Embedding(self.max_dist, embedding_dim)
        self.temporal_dist_embedding = nn.Embedding(self.max_dist, embedding_dim)
        self.WH = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.WS = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.WP = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.WQ = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

        # Decoder 部分
        self.constraint_mlp = nn.Linear(2 * embedding_dim, embedding_dim)
        self.concat_mlp = nn.Linear(3 * embedding_dim, embedding_dim)
        self.decoder = nn.LSTMCell(input_size=embedding_dim, hidden_size=embedding_dim)
        self.classify = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(embedding_dim, self.cell_nums),
            nn.Softmax(dim=-1)
        )

    def forward(self, src_ids, trg_ids=None, sample_ids=None, teacher_forcing_ratio=0.5):
        """
        :param src_ids: tensor: [batch_size, src_len]
        :param trg_ids: tensor: [batch_size, trg_len]
        :param sample_ids: list eg. [0, 1, 2, -1, 4, -1, -1, 7]
        :param teacher_forcing_ratio:
        :return:
        """
        batch_size, src_len = src_ids.shape[0], src_ids.shape[1]
        trg_len = len(sample_ids)

        src_ids = src_ids.to(self.device)
        src_ids = src_ids.long()
        if trg_ids is not None:
            trg_ids = trg_ids.long()
            trg_ids = trg_ids.to(self.device)
        # else:
        #     trg_ids = torch.zeros(size=(batch_size, trg_len)).to(device)

        precursor_ids, successor_ids = self.get_precursor_and_successor(sample_ids)
        src_time_steps = torch.tensor([idx for idx in sample_ids if idx != -1]).repeat(batch_size, 1)

        input_embedding = self.loc_embedding(src_ids)
        enc_s, (enc_h, enc_c) = self.encoder(input_embedding)

        output_prob = torch.zeros(size=(batch_size, trg_len, self.cell_nums))
        output_ids = torch.zeros(size=(batch_size, trg_len))
        output_ids[:, 0] = src_ids[:, 0]
        prev_point = self.loc_embedding(src_ids[:, 0])
        dec_h, dec_c = enc_s[:, -1, :], torch.zeros(size=(batch_size, enc_s.shape[2])).to(self.device)
        for i in range(1, len(sample_ids)):
            # dec_h 和 enc_s 计算 Attention
            # dec_h: [batch_size, embedding_dim]
            attn_h = dec_h.unsqueeze(1).repeat(1, src_len, 1)
            temporal_dist = torch.abs(torch.tensor(i).repeat(batch_size, src_len) - src_time_steps).long().to(self.device)
            temporal_dist = torch.where(temporal_dist >= self.max_dist, self.max_dist - 1, temporal_dist)
            dt = self.temporal_dist_embedding(temporal_dist)
            spatial_dist = self.cal_spatial_dist(output_ids[:, -1], src_ids).to(self.device)  # ds: (batch_size, seq_len)
            spatial_dist = torch.where(spatial_dist >= self.max_dist, self.max_dist - 1, spatial_dist)
            ds = self.spatial_dist_embedding(spatial_dist)

            wh = self.WH(attn_h)
            ws = self.WS(enc_s)
            wp = self.WP(dt)
            wq = self.WQ(ds)
            attn_u = self.v(torch.tanh(wh + ws + wp + wq)).squeeze(2)  # attn_u: (batch_size, src_len)
            attn_a = F.softmax(attn_u, dim=1)
            attn_e = torch.bmm(attn_a.unsqueeze(2).transpose(1, 2), enc_s).squeeze(1) # attn_e: (batch_size, embedding_size)

            pre_embedding = input_embedding[:, precursor_ids[i], :]
            suc_embedding = input_embedding[:, successor_ids[i], :]
            r = self.constraint_mlp(torch.cat((pre_embedding, suc_embedding), dim=1))
            dec_input = self.concat_mlp(torch.cat((prev_point, r, attn_e), dim=1))

            dec_h, dec_c = self.decoder(dec_input, (dec_h, dec_c))
            loc_prob = self.classify(dec_h) # loc_prob: (batch_size, cell_nums)
            loc_ids = torch.argmax(loc_prob, dim=1)
            if sample_ids[i] == -1:
                output_prob[:, i, :] = loc_prob
                output_ids[:, i] = loc_ids
            else:
                # TODO
                # one_hot = torch.zeros(size=(batch_size, self.cell_nums))
                # for j in range(one_hot.shape[0]):
                #     one_hot[j, trg_ids[j, i]] = 1
                output_ids[:, i] = trg_ids[:, i]

            if i == len(sample_ids) - 1:
                break
            if trg_ids is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                prev_point = self.loc_embedding(trg_ids[:, i + 1]) if teacher_force else self.loc_embedding(loc_ids)
            else:
                prev_point = self.loc_embedding(loc_ids)

        return output_prob, output_ids

    @staticmethod
    def get_precursor_and_successor(sample_ids):
        precursor_ids, successor_ids = [], []
        src_len = 0
        match = {}
        for cid in sample_ids:
            if cid != -1:
                match[cid] = src_len
                src_len += 1
        for i, cid in enumerate(sample_ids):
            j = i - 1
            while j > 0:
                if sample_ids[j] != -1:
                    break
                j -= 1
            pre = sample_ids[0] if j <= 0 else sample_ids[j]
            j = i + 1
            while j < len(sample_ids):
                if sample_ids[j] != -1:
                    break
                j += 1
            suc = sample_ids[-1] if j >= len(sample_ids) else sample_ids[j]
            precursor_ids.append(match[pre])
            successor_ids.append(match[suc])
        return precursor_ids, successor_ids

    def id_to_loc(self, cid):
        """
        把栅格 ID 转换为行列
        :param cid:
        :return: (r, c)
        """
        return cid // self.col_nums, cid % self.col_nums

    def cal_cell_dist(self, src_cell, trg_cell):
        """
        计算栅格距离，以栅格边长为单位
        :param src_cell:
        :param trg_cell:
        :return:
        """
        r1, c1 = self.id_to_loc(src_cell)
        r2, c2 = self.id_to_loc(trg_cell)
        return int(math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))

    def cal_spatial_dist(self, cur_cell_id, src_cell_ids):
        """
        :param cur_cell_id: (batch_size)
        :param src_cell_ids: (batch_size, seq_len)
        :return:
        """
        batch_size, seq_len = src_cell_ids.shape[0], src_cell_ids.shape[1]
        dist = torch.zeros(size=(batch_size, seq_len))
        for i in range(batch_size):
            u = cur_cell_id[i].item()
            for j in range(seq_len):
                v = src_cell_ids[i, j].item()
                dist[i, j] = self.cal_cell_dist(u, v)
        return dist.long()


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, trg_ids, recover_idx):
        sum_loss = torch.tensor(0.0)
        batch_idx = [k for k in range(x.shape[0])]
        for idx in recover_idx:
            sum_loss += torch.sum(-torch.log(x[batch_idx, idx, list(trg_ids[:, idx])]))
        avg_loss = sum_loss / (x.shape[0] * len(recover_idx))
        return avg_loss




