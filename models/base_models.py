import math

import numpy as np
import torch
import torch.nn as nn

import manifolds
import models.encoders as encoders
from utils.helper import default_device
from math import pow
from torch_geometric.utils import structured_negative_sampling


class HGCFModel(nn.Module):

    def __init__(self, users_items, args):
        super(HGCFModel, self).__init__()

        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, "HGCF")(self.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args

        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())

        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))

        self.embedding.weight = manifolds.ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)

    def encode(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
            adj = adj.to(default_device())
            x = x.to(default_device())
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return sqdist

    def temp(self, embeddings, neg_edges_list, pos_edges, num_neg, top_k):
        pass

    def compute_neg_distance(self, embeddings, neg_edges_list, pos_edges, epoch, cu_learn=False):
        pos_items = embeddings[pos_edges[:, 1], :]
        all_distance = torch.tensor([], device=default_device())
        all_neg_items = torch.tensor([], device=default_device())
        neg_num = len(neg_edges_list)

        for neg_edges in neg_edges_list:
            neg_item = embeddings[neg_edges[:, 1], :]
            distance = self.manifold.sqdist(neg_item, pos_items, self.c)
            all_distance = torch.cat((all_distance, distance), 1)
            all_neg_items = torch.cat((all_neg_items, neg_item), 1)

        _, indices = torch.sort(all_distance)
        if cu_learn is True:
            # index = math.floor(neg_num * (1 - epoch / self.args.epochs))
            index = int(neg_num * (1 - pow(epoch / self.args.epochs, 0.5)))
            # print(index)
            # if epoch < 20:
            #     index = 9
            # elif epoch < 40:
            #     index = 8
            # elif epoch < 60:
            #     index = 7
            # elif epoch < 80:
            #     index = 6
            # elif epoch < 100:
            #     index = 5
            # elif epoch < 150:
            #     index = 4
            # elif epoch < 200:
            #     index = 3
            # elif epoch < 300:
            #     index = 2
            # elif epoch < 400:
            #     index = 1
            # else:
            #     index = 0
            # if epoch < 100:
            #     index = 2
            # elif epoch < 300:
            #     index = 1
            # else:
            #     index = 0
            indices = indices[:, index].unsqueeze(1)
        else:
            indices = indices[:, 0].unsqueeze(1)

        vec_length = pos_items.shape[1]
        new_indices = torch.tensor([], device=default_device(), dtype=torch.long)
        for i in range(vec_length):
            new_indices = torch.cat((new_indices, indices * vec_length + i), 1)

        new_neg_items = torch.gather(all_neg_items, 1, new_indices)
        users = embeddings[pos_edges[:, 0], :]
        neg_scores = self.manifold.sqdist(users, new_neg_items, self.c)

        return neg_scores

    def compute_loss(self, embeddings, triples, epoch):
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)

        # neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
        #                    sampled_false_edges_list]
        # neg_scores = torch.cat(neg_scores_list, dim=1)
        neg_scores = self.compute_neg_distance(embeddings, sampled_false_edges_list, train_edges, epoch, cu_learn=True)

        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
