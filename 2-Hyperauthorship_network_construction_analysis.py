# name: Hyperauthorship_network_construction_analysis.py
# author: Ly Dinh
# date created: 01/18/2023
# note: replicate the same code for edgelist with/without hyperauthorship

import pandas as pd
!pip install --upgrade scipy networkx
!pip install signed_backbones
from networkx import *
import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter
import time
import pickle
import os
import math
!pip install powerlaw
import scipy.stats as stats
import pylab
import scipy
import seaborn as sns # good for visualizations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from statistics import mean
import random

file =“your path here"
paper_author = pd.read_csv(file, keep_default_na=False, encoding = 'unicode_escape')

# Add node attributes
nodelist= pd.read_csv(‘your file path here')
u2 = nodelist.select_dtypes(object)
nodelist[u2.columns] = u2.apply(
    lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

for i, nlrow in nodelist.iterrows():
  try:
    paper_author_net.nodes[nlrow['Target']].update(nlrow[1:].to_dict())
  except:
    continue
print(nlrow)

nodelist = nodelist.convert_dtypes()

# must be imported as nx.DiGraph to preserve order of source,target (paper, author)
paper_author_net = nx.from_pandas_edgelist(paper_author, source='Source', target='Target', edge_attr=True, create_using=nx.Graph())

target_nodelist = paper_author['Target']
target_nodelist = target_nodelist.drop_duplicates()
target_nodelist = target_nodelist.to_list()

source_nodelist = paper_author['Source']
source_nodelist = source_nodelist.drop_duplicates()
source_nodelist = source_nodelist.to_list()

projection_withouthyperauthor = bipartite.weighted_projected_graph(paper_author_net,target_nodelist, ratio=False)
nx.write_edgelist(projection_withouthyperauthor, "author-author_edgelist_without_hyperauthor.csv") ## new code 1/20/23

# Specify the output CSV file path
output_csv_file = "author-author_edgelist_without_hyperauthor.csv"

# Open the CSV file for writing
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row (optional)
    writer.writerow(["Source", "Target", "Weight"])

    # Iterate through edges and write them to the CSV file
    for edge in projection_withouthyperauthor.edges(data=True):
        node1, node2, data = edge
        weight = data["weight"]
        writer.writerow([node1, node2, weight])

number_of_nodes(projection_withouthyperauthor)

number_of_edges(projection_withouthyperauthor)

weighted_edgelist = nx.write_weighted_edgelist(projection_withouthyperauthor, "projection_withouthyperauthor_weighted.csv", delimiter='\t', encoding = 'utf-8')

## Community Detection
# Visualizing communities with (1) Clauset-Newman-Moore modularity maximization algorithm
from networkx.algorithms.community import *

community_CNM = greedy_modularity_communities(projection_withouthyperauthor)
len(community_CNM)

## Kernighan-lin algorithm
community_KL = kernighan_lin_bisection(projection_withouthyperauthor)
len(community_KL)

## Clique percolation
len(list(nx.community.k_clique_communities(projection_withouthyperauthor, 3)))

## Louvain modularity
community_L = louvain_communities(projection_withouthyperauthor)
len(community_L)

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(projection_withouthyperauthor.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(projection_withouthyperauthor.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

## Unweighted betwenness centrality
between_cent = nx.betweenness_centrality(projection_withouthyperauthor, weight=None,k=3000, normalized = True)
np.mean(list(between_cent.values()))

## Invert edge weights for betweenness, closeness, and APL
for u, v, weight in projection_withouthyperauthor.edges(data='weight'):
    projection_withouthyperauthor[u][v]['weight'] = 1 / weight

# Print the updated edge weights
for u, v, weight in projection_withouthyperauthor.edges(data='weight'):
    print(f'Edge ({u}, {v}) has an inverse weight of {weight:.2f}')

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(projection_withouthyperauthor, weight='weight',k=3000, normalized = True)
np.mean(list(between_cent_wt.values()))

# Weighted closeness centrality, using the edge weights as the cost of traversing each edge
close_cent_wt = nx.closeness_centrality(projection_withouthyperauthor, distance = 'weight', wf_improved = True)
np.mean(list(close_cent_wt.values()))

# Weighted eigenvector centrality
eig_cent_wt = nx.eigenvector_centrality(projection_withouthyperauthor, weight='weight')
eig_cent = nx.eigenvector_centrality(projection_withouthyperauthor)
print('Weighted eigenvector: {}'.format(np.mean(list(eig_cent_wt.values()))))
print('Unweighted eigenvector: {}'.format(np.mean(list(eig_cent.values()))))

#D ensity
print(nx.density(projection_withouthyperauthor))

# Clustering 
G_simple=nx.Graph(projection_withouthyperauthor)
print(nx.average_clustering(G_simple))

# APL
giantC = projection_withouthyperauthor.subgraph(max(nx.connected_components(projection_withouthyperauthor), key=len))
print(nx.average_shortest_path_length(giantC))

def random_subgraph(graph, N):
    nodes = random.sample(graph.nodes(), N)
    return graph.subgraph(nodes)

# Unweighted APL
subnet = random_subgraph(giantC, 6200) ## gets us 3000 nodes in giantC
giantC_subnet = subnet.subgraph(max(nx.connected_components(subnet), key=len))
number_of_nodes(giantC_subnet)
print(nx.average_shortest_path_length(giantC_subnet))

# Weighted APL
print(nx.average_shortest_path_length(giantC_subnet, weight = 'weight'))

# Connected components
print(nx.number_connected_components(projection_withouthyperauthor))

# Small-world coefficient
omega(giantC_subnet, niter=20, nrand=20, seed=1)

# Preferential attachment test
degree_sequence = sorted([d for n, d in projection_withouthyperauthor.degree()], reverse=True) # used for degree distribution and powerlaw test
import powerlaw # Power laws are probability distributions with the form:p(x)∝x−α
fit = powerlaw.Fit(degree_sequence)
fit.power_law.alpha

## Create Ego-graphs
ID_egonet = ego_graph(projection_withouthyperauthor, 'ID', radius=1, center=True, undirected=False, distance='Weight') # Include all neighbors of distance<=radius from n.

## network measures for egonet
print('Average Degree: {}'.format(2*ID_egonet.number_of_edges() / (ID_egonet.number_of_nodes())))
print('Density: {}'.format(nx.density(ID_egonet)))
print('Average shortest path (geodesic) length: {}'.format(nx.average_shortest_path_length(ID_egonet)))
print('Average clustering: {}'.format(nx.average_clustering(ID_egonet)))
print('Num of nodes: {}'.format(nx.number_of_nodes(ID_egonet)))
print('Num of edges: {}'.format(nx.number_of_edges(ID_egonet)))

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(ID_egonet.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(ID_egonet.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(ID_egonet, weight='weight', normalized = True)
print('Weighted betweenness: {}'.format(np.mean(list(between_cent_wt.values()))))
between_cent = nx.betweenness_centrality(ID_egonet, weight=None, normalized = True)
print('Unweighted betweenness: {}'.format(np.mean(list(between_cent.values()))))

# Weighted eigenvector centrality
eigen_cent_wt = nx.eigenvector_centrality(ID_egonet, weight='weight')
print('Weighted eigenvector: {}'.format(np.mean(list(eigen_cent_wt.values()))))
eigen_cent = nx.eigenvector_centrality(ID_egonet, weight=None)
print('Unweighted eigenvector: {}'.format(np.mean(list(eigen_cent.values()))))

# Weighted closeness centrality
closeness_cent_wt = nx.closeness_centrality(ID_egonet, distance='weight')
print('Weighted closeness: {}'.format(np.mean(list(closeness_cent_wt.values()))))
closeness_cent = nx.closeness_centrality(ID_egonet, distance=None)
print('Unweighted closeness: {}'.format(np.mean(list(closeness_cent.values()))))

# Plot egonet
plt.figure(figsize =(10, 10))
degree = dict(nx.degree_centrality(ID_egonet))
color_map = []
for node in ID_egonet:
    if node == 'Author Name':
        color_map.append('steelblue')
    else:
        color_map.append('navy')
nx.draw(ID_egonet, with_labels = False, nodelist=degree.keys(),
        node_size=[s * 1000 for s in degree.values()], node_color = color_map,
        edge_color='lightsteelblue')
plt.title("Egonetwork of ID: Without hyperauthorship")
plt.savefig("ID_withouthyperauthorauthor_egonet.png")

# Count all # of co authors
coauthors_withouthyperauthor = sorted(list(projection_withouthyperauthor.degree))
coauthors_withouthyperauthor_df = pd.DataFrame(coauthors_withouthyperauthor, columns = ['author','num_coauthors'])
coauthors_withouthyperauthor_df.to_csv("coauthors_withouthyperauthor.csv")

# Adding new column to count for number of authors with k num of co-authors
num_coauthors = collections.Counter(coauthors_withouthyperauthor_df['num_coauthors'])
numcoauthors_df = pd.DataFrame.from_dict(num_coauthors, orient='index').reset_index()
numcoauthors_df = numcoauthors_df.rename({'index':'num_coauthors',0:'num_authors_with_coauthor'}, axis=1)
numcoauthors_df = numcoauthors_df.sort_values(by='num_coauthors', ascending=True)

## make plot
plt.figure(figsize=(10,8))
plt.plot(numcoauthors_df['num_coauthors'], numcoauthors_df['num_authors_with_coauthor'], color='black', marker='o')
plt.xlabel('Number of co-authors (per author)', fontsize=10)
plt.ylabel('Number of authors with k co-authors', fontsize=10)
plt.title('Without hyperauthorship', fontsize=12)
plt.grid(True)
plt.show()

# Log transformation of plot; make our highly-skewed distribution less skewed
plt.rcParams.update({'font.size': 22})

numcoauthors_log = np.log(numcoauthors_df)
plt.figure(figsize=(10,8))
plt.plot(numcoauthors_log['num_coauthors'], numcoauthors_log['num_authors_with_coauthor'], color='black', marker='o')
plt.xlabel('Number of co-authors (per author)', fontsize=20)
plt.ylabel('Number of authors with k co-authors', fontsize=20)
plt.title('without hyperauthors', fontsize=12)
plt.grid(True)
plt.savefig('coauthor_dist_withouthyperauthor_log.pdf')
plt.show()

# Function to return plots for the feature of checking normality to see that lognormal transformation is a good fit
def normality(data,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(data[feature])
    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=pylab)
    plt.show()

normality(numcoauthors_log,'num_coauthors') ## looks more 'normal' now

degreeView = paper_author_net.degree(source_nodelist)
degreeView = pd.DataFrame(degreeView, columns=['paper_name','num_coauthors'])
degreeView.to_csv('paper_counts_withouthyperauthor.csv',header=['paper_name','num_coauthors'])

# Count number of authors per paper & count how many papers with # of authors
num_coauthors_per_paper = collections.Counter(degreeView['num_coauthors'])
num_coauthors_per_paper = pd.DataFrame.from_dict(num_coauthors_per_paper, orient='index').reset_index()
num_coauthors_per_paper = num_coauthors_per_paper.rename({'index':'num_coauthors',0:'papers_with_num_coauthors'}, axis=1)
num_coauthors_per_paper = num_coauthors_per_paper.sort_values(by='num_coauthors', ascending=True)

# Make plot
plt.figure(figsize=(10,8))
plt.plot(num_coauthors_per_paper['num_coauthors'], num_coauthors_per_paper['papers_with_num_coauthors'], color='black', marker='o')
plt.xlabel('Number of co-authors (per paper)', fontsize=20)
plt.ylabel('Number of papers with k co-authors', fontsize=20)
plt.title('Without hyperauthors', fontsize=12)
plt.grid(True)
plt.show()

# Log transformation of plot; make our highly-skewed distribution less skewed

num_coauthors_per_paper_log = np.log(num_coauthors_per_paper)
plt.figure(figsize=(10,8))
plt.plot(num_coauthors_per_paper_log['num_coauthors'], num_coauthors_per_paper_log['papers_with_num_coauthors'], color='black', marker='o')
plt.xlabel('Number of co-authors (per paper)', fontsize=20)
plt.ylabel('Number of papers with k co-authors', fontsize=20)
plt.title('without hyperauthors', fontsize=12)
plt.grid(True)
plt.savefig('papers_authors_dist_withouthyperauthor_log.pdf')
plt.show()

#function to return plots for the feature of checking normality to see that lognormal transformation is a good fit
def normality(data,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(data[feature])
    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=pylab)
    plt.show()

normality(num_coauthors_per_paper_log,'num_coauthors') ## looks more 'normal' now

degreeView.to_csv("withouthyperauthor_unique papers & authors counts.csv")

## Fractional Counting (Newman & Jaccard)

## Newman weighted projection
projection_newman_withouthyperauthor = bipartite.collaboration_weighted_projected_graph(paper_author_net,target_nodelist)
#list(projection_withouthyperauthor.edges(data=True))
nx.write_edgelist(projection_newman_withouthyperauthor, "author-author_edgelist_newman_without_hyperauthor.csv") 

number_of_nodes(projection_newman_withouthyperauthor)

number_of_edges(projection_newman_withouthyperauthor)

nx.write_weighted_edgelist(projection_newman_withouthyperauthor, "projection_withouthyperauthor_weighted_newman.csv", delimiter = ',')
weighted_edgelist = pd.read_csv('projection_withouthyperauthor_weighted_newman.csv',names=["Source","Target","Weight"])
weighted_edgelist.to_csv('projection_withouthyperauthor_weighted_newman.csv')

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(projection_newman_withouthyperauthor.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(projection_newman_withouthyperauthor.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

# Unweighted betwenness centrality
between_cent = nx.betweenness_centrality(projection_newman_withouthyperauthor, weight=None,k=3000, normalized = True)
np.mean(list(between_cent.values()))

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(projection_newman_withouthyperauthor, weight='weight',k=3000, normalized = True)
np.mean(list(between_cent_wt.values()))

# Weighted closeness centrality, using the edge weights as the cost of traversing each edge
close_cent_wt = nx.closeness_centrality(projection_newman_withouthyperauthor, distance = 'weight')
np.mean(list(close_cent_wt.values()))

# Weighted eigenvector centrality
eig_cent_wt = nx.eigenvector_centrality(projection_newman_withouthyperauthor, weight='weight')
eig_cent = nx.eigenvector_centrality(projection_newman_withouthyperauthor)
print('Weighted eigenvector: {}'.format(np.mean(list(eig_cent_wt.values()))))
print('Unweighted eigenvector: {}'.format(np.mean(list(eig_cent.values()))))

# Create Ego-graph
ID_egonet_newman = ego_graph(projection_newman_withouthyperauthor, 'ID', radius=1, center=True, undirected=False, distance='Weight') # Include all neighbors of distance<=radius from n.

# Network measures for egonet
print('Average Degree: {}'.format(2*ID_egonet_newman.number_of_edges() / (ID_egonet_newman.number_of_nodes())))
print('Density: {}'.format(nx.density(ID_egonet_newman)))
print('Average shortest path (geodesic) length: {}'.format(nx.average_shortest_path_length(ID_egonet_newman)))
print('Average clustering: {}'.format(nx.average_clustering(ID_egonet_newman)))

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(ID_egonet_newman.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(ID_egonet_newman.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(ID_egonet_newman, weight='weight', normalized = True)
print('Weighted betweenness: {}'.format(np.mean(list(between_cent_wt.values()))))
between_cent = nx.betweenness_centrality(ID_egonet_newman, weight=None, normalized = True)
print('Unweighted betweenness: {}'.format(np.mean(list(between_cent.values()))))

# Weighted eigenvector centrality
eigen_cent_wt = nx.eigenvector_centrality(ID_egonet_newman, weight='weight')
print('Weighted eigenvector: {}'.format(np.mean(list(eigen_cent_wt.values()))))
eigen_cent = nx.eigenvector_centrality(ID_egonet_newman, weight=None)
print('Unweighted eigenvector: {}'.format(np.mean(list(eigen_cent.values()))))

# Weighted closeness centrality
closeness_cent_wt = nx.closeness_centrality(ID_egonet_newman, distance='weight')
print('Weighted closeness: {}'.format(np.mean(list(closeness_cent_wt.values()))))
closeness_cent = nx.closeness_centrality(ID_egonet_newman, distance=None)
print('Unweighted closeness: {}'.format(np.mean(list(closeness_cent.values()))))

# Plot egonet
plt.figure(figsize =(10, 10))
degree = dict(nx.degree_centrality(ID_egonet_newman))
color_map = []
for node in ID_egonet_newman:
    if node == 'Author Name':
        color_map.append('steelblue')
    else:
        color_map.append('navy')
nx.draw(ID_egonet_newman, with_labels = False, nodelist=degree.keys(),
        node_size=[s * 1000 for s in degree.values()], node_color = color_map,
        edge_color='lightsteelblue')
plt.title("Egonetwork of ID: Without hyperauthorship")
plt.savefig("ID_withouthyperauthorauthor_egonet_newman.png")

## Jaccard weighted projection
projection_jaccard_withouthyperauthor = bipartite.overlap_weighted_projected_graph(paper_author_net,target_nodelist)
#list(projection_withouthyperauthor.edges(data=True))
nx.write_edgelist(projection_jaccard_withouthyperauthor, "author-author_edgelist_jaccard_with_hyperauthor.csv")

number_of_nodes(projection_jaccard_withouthyperauthor)

number_of_edges(projection_jaccard_withouthyperauthor)

nx.write_weighted_edgelist(projection_jaccard_withouthyperauthor, "projection_withouthyperauthor_weighted_jaccard.csv", delimiter = ',')
weighted_edgelist = pd.read_csv('projection_withouthyperauthor_weighted_jaccard.csv',names=["Source","Target","Weight"])
weighted_edgelist.to_csv('projection_withouthyperauthor_weighted_jaccard.csv')

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(projection_jaccard_withouthyperauthor.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(projection_jaccard_withouthyperauthor.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

# Unweighted betwenness centrality
between_cent = nx.betweenness_centrality(projection_jaccard_withouthyperauthor, weight=None,k=3000, normalized = True)
np.mean(list(between_cent.values()))

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(projection_jaccard_withouthyperauthor, weight='weight',k=3000, normalized = True)
np.mean(list(between_cent_wt.values()))

# Weighted closeness centrality, using the edge weights as the cost of traversing each edge
close_cent_wt = nx.closeness_centrality(projection_jaccard_withouthyperauthor, distance = 'weight')
np.mean(list(close_cent_wt.values()))

# Weighted eigenvector centrality
eig_cent_wt = nx.eigenvector_centrality(projection_jaccard_withouthyperauthor, weight='weight')
eig_cent = nx.eigenvector_centrality(projection_jaccard_withouthyperauthor)
print('Weighted eigenvector: {}'.format(np.mean(list(eig_cent_wt.values()))))
print('Unweighted eigenvector: {}'.format(np.mean(list(eig_cent.values()))))

# Create Ego-graph 
ID_egonet_jaccard = ego_graph(projection_jaccard_withouthyperauthor, 'ID', radius=1, center=True, undirected=False, distance='Weight') # Include all neighbors of distance<=radius from n.

# Network measures for egonet for ID network
print('Average Degree: {}'.format(2*ID_egonet_jaccard.number_of_edges() / (ID_egonet_jaccard.number_of_nodes())))
print('Density: {}'.format(nx.density(ID_egonet_jaccard)))
print('Average shortest path (geodesic) length: {}'.format(nx.average_shortest_path_length(ID_egonet_jaccard)))
print('Average clustering: {}'.format(nx.average_clustering(ID_egonet_jaccard)))

# Weighted vs unweighted degree centrality
deg_cent_wt = dict(ID_egonet_jaccard.degree(weight = 'weight')).values()
deg_cent_wt = list(deg_cent_wt)
np.mean(list(deg_cent_wt))
deg_cent = dict(ID_egonet_jaccard.degree()).values()
deg_cent = list(deg_cent)
print('Weighted degree: {}'.format(np.mean(list(deg_cent_wt))))
print('Unweighted degree: {}'.format(np.mean(list(deg_cent))))

# Weighted betweenness centrality
between_cent_wt = nx.betweenness_centrality(ID_egonet_jaccard, weight='weight', normalized = True)
print('Weighted betweenness: {}'.format(np.mean(list(between_cent_wt.values()))))
between_cent = nx.betweenness_centrality(ID_egonet_jaccard, weight=None, normalized = True)
print('Unweighted betweenness: {}'.format(np.mean(list(between_cent.values()))))

# Weighted eigenvector centrality
eigen_cent_wt = nx.eigenvector_centrality(ID_egonet_jaccard, max_iter=500, weight='weight')
print('Weighted eigenvector: {}'.format(np.mean(list(eigen_cent_wt.values()))))
eigen_cent = nx.eigenvector_centrality(ID_egonet_jaccard, weight=None)
print('Unweighted eigenvector: {}'.format(np.mean(list(eigen_cent.values()))))

# Weighted closeness centrality
closeness_cent_wt = nx.closeness_centrality(ID_egonet_jaccard, distance='weight')
print('Weighted closeness: {}'.format(np.mean(list(closeness_cent_wt.values()))))
closeness_cent = nx.closeness_centrality(ID_egonet_jaccard, distance=None)
print('Unweighted closeness: {}'.format(np.mean(list(closeness_cent.values()))))

# Plot egonet
plt.figure(figsize =(10, 10))
degree = dict(nx.degree_centrality(ID_egonet_jaccard))
color_map = []
for node in ID_egonet_jaccard:
    if node == 'Author Name':
        color_map.append('darkblue')
    else:
        color_map.append('lightblue')
nx.draw(ID_egonet_jaccard, with_labels = False, nodelist=degree.keys(), node_size=[s * 1000 for s in degree.values()], node_color = color_map)
plt.title("Egonetwork of ID: Without hyperauthorship")
plt.savefig("ID_withhyperauthorauthor_jaccard_egonet.png")
