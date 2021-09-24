# -*- encoding: utf-8 -*-
"""
@Project    :   HGCF_oral
@File       :   test.py
@Time       :   2021/9/4 16:20
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
import torch
import numpy as np
from torch_geometric.utils import structured_negative_sampling


def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# min_index = torch.from_numpy(np.array([0, 1, 2]))
# neg_item_1 = torch.from_numpy(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
# neg_item_2 = torch.from_numpy(np.array([[11.0, 12.0], [12.0, 13.0], [13.0, 14.0]]))
# neg_item_3 = torch.from_numpy(np.array([[111.0, 112.0], [112.0, 113.0], [113.0, 114.0]]))
# value_mt = torch.from_numpy(np.array([[0, 1, 2], [3, 5, 4], [8, 7, 6]]))
# value_mt, min_mt = torch.sort(value_mt)
# print(value_mt)
# print(min_mt)
# min_index = min_mt[:, 0]
# print(min_index)
# min_index = torch.cat((min_index*2, min_index*2+1), dim=-1)
# print(min_index)
#
# neg_item_list = [neg_item_1, neg_item_2, neg_item_3]
# print(neg_item_list)
#
# all_items = torch.tensor([], dtype=torch.float64)
# for neg_item in neg_item_list:
#     all_items = torch.cat((all_items, neg_item), dim=1)
# new_item = torch.gather(all_items, 1, min_index)
# print(new_item)

edge_index = torch.tensor([[0, 1], [1, 0],
                           [1, 2], [2, 1],
                           [1, 3], [3, 1],
                           [2, 4], [4, 2]])

seed = 1234
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

sample_list = list()
for i in range(2):
    sample = structured_negative_sampling(edge_index.t().contiguous())[-1].tolist()
    sample_list.append(sample)
samples = np.array(sample_list)
print(samples.reshape((samples.shape[1], samples.shape[0])))
