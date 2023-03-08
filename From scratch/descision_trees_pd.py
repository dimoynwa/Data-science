import numpy as np
import pandas as pd

def column_entropy(df, col):
    total_count = df.shape[0] # number of rows
    unique_values = df[col].unique()
    # print(f'Unique values in {col}: {unique_values}')
    entropy = 0
    for val in unique_values:
        count = df.loc[df[col] == val].shape[0]
        # print(f'Value {val} and count {count} total count {total_count} and log {np.log2(count / total_count)}')
        entropy -= count / total_count * np.log2(count / total_count)
    return entropy

def information_gain(df, column, target_column):
    entropy = column_entropy(df, target_column)
    total = df.shape[0]
    
    unique_values = np.unique(df[column])
    for val in unique_values:
        val_df = df.loc[df[column] == val]
        count = val_df.shape[0]
        val_entropy = column_entropy(val_df, target_column)
        
        entropy -= count / total * val_entropy
    return entropy

class Node:
    def __init__(self, feature=None, children=None, value=None) -> None:
        self.feature = feature
        self.children = children # dictionary feature_value: node
        self.value = value

    def is_leaf(self):
        return self.value
    
    def visualize(self, level=0):
        if self.is_leaf():
            print('\t'*level, self.value)
            return
        print('\t'*level, self.feature, ':', '-->')
        if not self.children:
            return
        for key in self.children:
            print('\t'*(level+1), key, '-->')
            node = self.children[key]
            node.visualize(level=level+2)

class DescisionTree:
    def __init__(self, target_column, csv_file=None, df=None, max_depth=100, min_samples_split=2):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file)
        self.target_column = target_column
        self.features = [col for col in self.df.columns if col != target_column]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self):
        self.root = self._grow_tree(self.df, features=self.features)
    
    # Recursive function, which returns the root of the tree
    def _grow_tree(self, df, features, current_depth=0):
        # Check for stop conditions
        unique = np.unique(df[self.target_column])
        if len(unique) == 1:
            print(f'Only {unique} value left')
            return Node(value=unique[0])
        if current_depth == self.max_depth:
            print(f'Max depth reached: {current_depth}')
            return Node(value=df[self.target_column].value_counts().idxmax())
        if df.shape[0] <= self.min_samples_split:
            print(f'Min sample split is {self.min_samples_split}, but rows count is {df.shape[0]}')
            return Node(value=df[self.target_column].value_counts().idxmax())
        # Calculate Information Gain for all the features
        igs = np.array([information_gain(df, feat, self.target_column) for feat in features])
        print(f'Information gains: ', igs)
        max_gain_feature = features[igs.argmax()]
        print(f'Max gain feature: {max_gain_feature} from {features}')
        unique_values = np.unique(df[max_gain_feature])
        # Initializing childrens
        new_features = [f for f in features if f != max_gain_feature]
        print(f'New features: {new_features}')
        children = {val:self._grow_tree(df.loc[df[max_gain_feature] == val], \
                                      features=new_features, \
                                      current_depth=current_depth+1) for val in unique_values}
        return Node(feature=max_gain_feature, children=children)
    
    def predict(self, example):
        return self._predict(example, self.root)
    
    def visualize(self):
        self.root.visualize()

    def _predict(self, example, node: Node):
        if node.is_leaf():
            return node.value
        
        next_node = node.children.get(example[node.feature])
        if not next_node:
            print(f'Cannot classify example. Unknown value for feature {example[node.feature]}')
            return None
        return self._predict(example, next_node)
        
if __name__ == '__main__':
    tree = DescisionTree('Play Tennis', csv_file='/home/dimo/python-code/From scratch/PlayTennis.csv')
    tree.fit()

    tree.visualize()

    for idx in range(tree.df.shape[0]):
        example = tree.df.iloc[idx]
        prediction = tree.predict(example)
    
        expected = example[tree.target_column]
        print(f'Expected: {expected}, actual: {prediction}')