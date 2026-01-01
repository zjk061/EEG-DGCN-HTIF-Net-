# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from models.BaseModel import *

def get_context_feature(feed_dict, index, corpus, data):
	"""
	Get context features for the feed_dict, including user, item, and situation context
 	"""
	for c in corpus.user_feature_names:
		feed_dict[c] = corpus.user_features[feed_dict['user_id']][c]
	for c in corpus.situation_feature_names:
		feed_dict[c] = data[c][index]
	for c in corpus.item_feature_names:
		if type(feed_dict['item_id']) in [int, np.int32, np.int64]: # for a single item
			feed_dict[c] = corpus.item_features[feed_dict['item_id']][c]
		else: # for item list
			feed_dict[c] = np.array([corpus.item_features[iid][c] for iid in feed_dict['item_id']])
	return feed_dict

class ContextCTRModel(CTRModel):
	# context model for CTR prediction tasks
	reader = 'ContextReader'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.context_features = corpus.user_feature_names + corpus.item_feature_names + corpus.situation_feature_names\
					+ ['user_id','item_id']
		self.feature_max = corpus.feature_max

	class Dataset(CTRModel.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			feed_dict = get_context_feature(feed_dict, index, self.corpus, self.data)
			return feed_dict
