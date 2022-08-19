import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, min_samples_split):
        self.min_samples_split = min_samples_split
        self.tree_struct = dict()
        
    def entropy(self,node):
        groups, counts = np.unique(node['label'], return_counts = True)
        entropy = 0
        for i in range(len(groups)):
            entropy = entropy + (-counts[i]/np.sum(counts)*np.log2(counts[i]/np.sum(counts)))
        return np.round(entropy,4)
        
    def pick_feature(self,feat_list,node):
        base_entropy = self.entropy(node)
        feat_entropies = list()
        for feat in feat_list:
            levels, counts = np.unique(node[feat], return_counts = True)
            feat_entropy = 0 
            for i in range(len(levels)):
                subnode = node.loc[node[feat] == levels[i],:]
                feat_entropy = feat_entropy + counts[i]/np.sum(counts)*self.entropy(subnode)
            feat_entropies.append(feat_entropy)
        info_gain = base_entropy - np.array(feat_entropies)
        return feat_list[np.argmax(info_gain)]        
            
    def split(self,nodes, feat_list, level):

        self.tree_struct['level_'+str(level)+'_nodes'] = nodes

        if (len(nodes) == 0) or (len(feat_list[0]) == 0):
            print('---splitting finished!---')
            return

        new_nodes = []
        new_feat_list = []

        for i in range(len(nodes)):
            if (nodes[i]['status'] == 'splittable'):
                if (len(nodes[i]['data']) > self.min_samples_split):
                    node_feat = feat_list[i].copy()

                    fe = self.pick_feature(node_feat,nodes[i]['data'])
                    nodes[i]['feature'] = fe
                    feat_vals = np.unique(nodes[i]['data'].loc[:,fe])
                    node_feat.remove(fe)

                    subnodes = []
                    subnode_feat_list = []
                    n = {}


                    for val in feat_vals:    
                        n['branch_value'] = val
                        n['data'] = nodes[i]['data'].loc[nodes[i]['data'][fe] == val,:]
                        n['head_node'] = fe
                        n['status'] = 'splittable'
                        if (len(np.unique(n['data']['label'])) == 1):
                            n['status'] = 'pure'
                        subnodes.append(n)
                        subnode_feat_list.append(node_feat)
                        n = {}

                    new_nodes = new_nodes + subnodes
                    new_feat_list = new_feat_list + subnode_feat_list 
                else:
                    nodes[i]['status'] = 'end'


        return self.split(new_nodes,new_feat_list,level + 1) 

    def parse_tree(self,tree_struct):
    
        level = []
        feature = []
        state = []
        num_class = len(np.unique(self.tree_struct['level_0_nodes'][0]['data']['label']))

        group_distribution = []

        for key, item in self.tree_struct.items():
            level = level + [key[:-6]]*len(item)
            for node in item:
                state.append(node['status'])
                labels, counts = np.unique(node['data']['label'], return_counts = True)
                #if len(counts) < num_class:
                dummy_str = ['0']*num_class
                label_list = labels.tolist() 
                for i in range(len(label_list)):
                    dummy_str[label_list[i]] = counts[i]
                group_distribution.append('|'.join([str(el) for el in dummy_str]))
                #else:


                if 'feature' in node:
                    feature.append(node['feature'])
                else:
                    feature.append('')


        return pd.DataFrame({'Level':level, 'Feature':feature,'State':state,'Group distribution':group_distribution}) 
    
    def predict(self,dd):
        T = len(dd)
        preds = []
        d = 1
        n = 0
        head_node = self.tree_struct['level_0_nodes'][0]

        for i in range(0,len(dd)):
            example = dd.loc[i,:]
            node_found = False
            while not node_found:
                current_node = self.tree_struct['level_' + str(d) + '_nodes'][n]

                if (current_node['branch_value'] == example[head_node['feature']]):
                    if (current_node['status'] == 'splittable'):

                        if ('feature' in current_node):

                            d = d + 1
                            head_node = current_node
                            current_node = self.tree_struct['level_' + str(d) + '_nodes'][n]
                            n = 0
                        else:
                            preds.append(np.sum(current_node['data']['label'])/len(current_node['data']['label']))
                            d = 1
                            n = 0
                            head_node = self.tree_struct['level_0_nodes'][0]
                            node_found = True
                    else:
                        preds.append(np.sum(current_node['data']['label'])/len(current_node['data']['label']))
                        d = 1
                        n = 0
                        head_node = self.tree_struct['level_0_nodes'][0]
                        node_found = True
                else:
                    n = n + 1

        return np.round(np.array(preds),0).astype(int)
    
    def fit(self,dataframe):
        self.split([{'branch_value':None, 'data':dataframe, 'head_node':None, 'status':'splittable'}],
                   [dataframe.columns.tolist()[:-1]],0)

# Data

empStatus = ['Self Employed', 'Self Employed','Employed','Student/Postdoc.','Student/Postdoc.','Student/Postdoc.',
            'Employed','Self Employed','Self Employed','Student/Postdoc.','Self Employed','Employed','Employed',
             'Student/Postdoc.', 'Employed','Student/Postdoc.','Self Employed']
degree = ['Postgraduate','Postgraduate','Postgraduate','Postgraduate','Undergraduate','Undergraduate','Undergraduate',
         'Postgraduate','Undergraduate','Undergraduate','Undergraduate','Postgraduate','Undergraduate','Postgraduate',
          'Postgraduate','Postgraduate','Postgraduate']

cqf = ['no','yes','yes','no','yes','yes','yes','no','no','yes','yes','yes','no','yes','no','no','no']
subscriber = [0,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,0]

dd = pd.DataFrame({'empStatus':pd.Categorical(empStatus),'degree':pd.Categorical(degree),
                   'cqf':pd.Categorical(cqf),'label':subscriber})

#Fit the tree to data

dt = DecisionTree(min_samples_split = 2)
dt.fit(dd)
print('Accuracy: {}'.format(np.round(accuracy_score(dd['label'],dt.predict(dd)), 2)))

# parse DT as dataframe (rows are nodes)

dt.parse_tree(dt.tree_struct)
