from descision_trees_pd import DescisionTree

import pandas as pd

from collections import Counter

# Steps
# 1. Get a subset of the dataset
# 2. Create a Descision tree
# 3. Repeat for as many times as the number of trees

# Testing
# 1. Get a prediction from each tree
# 2.1. Classification: Get the majority vote
# 2.2. Regression: Get the mean of the prediction

class RandomForest:
    def __init__(self, csv_file, target_column, n_trees=10, sample_size=10, max_depth=100, min_samples_split=2) -> None:
        self.df = pd.read_csv(csv_file)
        self.n_trees = n_trees
        self.target_column = target_column
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size

    def fit(self):
        self.trees = []
        for _ in range(self.n_trees):
            rand_sample = self.df.sample(self.sample_size)
            print(rand_sample.index)
            tree = DescisionTree(target_column=self.target_column, df=rand_sample, max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split)
            tree.fit()
            self.trees.append(tree)
        
    def predict(self, example):
        predictions = [tree.predict(example) for tree in self.trees]
        print('Predictions: ', predictions)
        counter = Counter(predictions)
        return counter.most_common()
    
if __name__ == '__main__':
    forest = RandomForest('/home/dimo/python-code/From scratch/PlayTennis.csv', 'Play Tennis', n_trees=4)
    forest.fit()

    for tree in forest.trees:
        print('Tree:')
        tree.visualize()

    for idx in range(forest.df.shape[0]):
        example = forest.df.iloc[idx]
        prediction = forest.predict(example)
    
        expected = example[forest.target_column]
        print(f'Expected: {expected}, actual: {prediction}')
