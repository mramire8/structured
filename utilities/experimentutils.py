from sklearn.datasets.base import Bunch
from sklearn.datasets import base as bunch


def sample_data(data, train_idx, test_idx):
	sample = Bunch()
	sample.train.data = data.data[train_idx]
	sample.test.data = data.data[test_idx]
	sample.train.target = data.target[train_idx]
	sample.test.target = data.target[test_idx]
	sample.target_names = data.target_names

	return sample

def get_vectorizer(config):
	limit = config['limit']
	vectorizer = config['vectorizer']
	min_size = config['min_size']

	if vectorizer == 'tfidf':
		from sklearn.feature_extraction.text import  TfidfVectorizer
		return TfidfVectorizer(encoding='ISO-8859-1', min_df=5, max_df=1.0, binary=False, ngram_range=(1, 1))
	else:
		return None