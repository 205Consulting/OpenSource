import networkx as nx
from networkx.algorithms import bipartite
import gensim
import numpy as np

# todo:
# add hyperparameters i.e num_topics, iterations, w/e


### README:
# Requirements: each node in bipartite graph must have str() method
# and that str() method must be unique. needed for gensim (i think).
class bpg_recommender(object):
	# b: bipartite graph to run recommendations on
	def __init__(self, b, num_clusters = 6):
		self.b = b
		self.num_clusters = num_clusters
		# split the graph into it's two partitions
		self.nodes = list(bipartite.sets(b))
		self.mappings = {}

		# NOTE: self.copora[0] consideres each node in self.nodes[0] and makes a bag of songs representation for it's neighbors.
		# i.e self.corpora[0] is what we pass into lda when we want to model the nodes in self.nodes[0] as documents and the nodes in self.nodes[1] as "words"

		self.corpora, self.dicts = self._get_graph_corpora()
		# lda_models[0] would be the lda model where the "documents" are sets of nodes in self.nodes[1]
		self.lda_models = self._train_lda_models()
		# per_cluster_node_distributions[0] is an array of dicts, each dict mapping id->probability for that cluster index
		self.per_cluster_node_distributions = self._find_per_cluster_node_distributions()

		self.per_dnode_cluster_distributions = self._find_per_dnode_cluster_distributions()


	def additional_feature_factory(self, node):
		'''
			function: additional_feature_factory

			params: node - node to get features for

			returns: python list of additional features

			notes: see feature_factory. users should overrwrite this function.
		'''

		return []



	def feature_factory(self, node):
		'''
			function: feature_factory

			params: node - node to get features for

			returns: python list of features for the node

			notes: users that inherit bpg_recommender should overrwrite "additional_feature_factory" with 
			and specific features they have, and then call feature_factory on their nodes
		'''
		features = []
		# append all the base features that we have
		for mapping in mappings:
			features.append(mapping[str(node)])
		# append additional, non-agnostic features
		features += self.additional_feature_factory(node)




	def _find_per_dnode_cluster_distributions(self):
		return [self._find_per_dnode_cluster_distribution(0),self._find_per_dnode_cluster_distribution(1)]


	def _find_per_dnode_cluster_distribution(self, partition_index):
		return None
		#todo: not clear if this is something I should pre-compute.. not too hard to get from
		# gensim using .inference (as in the code currently)

	def _find_per_cluster_node_distributions(self):
		return [self._find_per_cluster_node_distribution(0), self._find_per_cluster_node_distribution(1)]

	def _find_per_cluster_node_distribution(self, partition_index):
		'''
			function: _find_per_cluster_node_distribution

			params: partition_index - which lda model we're looking at

			returns: an num-clusters length array, the ith element a dict representing the ith cluster. that dict
			has nodes as keys and probability that that cluster generates that node as values.

			notes: this function is essentially because gensim makes it wierd to find these values using their
			data structures
		'''


		dist = []
		# iterate through topics
		for cluster in range(self.num_clusters): 
			cluster_dist_dict = {}
			# get probability distribution
			cluster_dist = self.lda_models[partition_index].state.get_lambda()[cluster] 
			# normalize to real probability distribution
			cluster_dist = cluster_dist / cluster_dist.sum()
			for i in range(len(cluster_dist)):
				# map the string id of the node to the probability (self.dict goes from gensim's id -> my string id)
				cluster_dist_dict[self.dicts[partition_index][i]] = cluster_dist[i] 
			# append to array of dicts
			dist.append(cluster_dist_dict) 
		return dist
			


	def _get_partition_corpus(self,partition):
		'''
			function: _get_partition_corpus

			params: partition - which partition to get the corpus for

			returns: 	corpus - gensim corpus, each document being the neighbors of a node in our partition
						dictionary - gensim dictionary for the above corpus
		'''



		# build bags of neighbors
		bags_of_neighbors = []
		for node in partition:
			bags_of_neighbors.append(self._node_to_bagofneighbors(node))

		# create gensim dictionary
		dictionary = gensim.corpora.Dictionary(bags_of_neighbors)

		# change to wierd gensim bow format
		corpus = [dictionary.doc2bow(bon) for bon in bags_of_neighbors]

		return corpus, dictionary




	def _get_graph_corpora(self):
		# left nodes:
		zero_corpus, zero_dict = self._get_partition_corpus(self.nodes[0])

		# right nodes:
		one_corpus, one_dict = self._get_partition_corpus(self.nodes[1])

		return [zero_corpus, one_corpus], [zero_dict, one_dict]

	

	def _train_lda_models(self):
		return [self._train_lda_model(0), self._train_lda_model(1)]

	def _train_lda_model(self, partition_index):
		lda = gensim.models.ldamodel.LdaModel(corpus=self.corpora[partition_index], id2word=self.dicts[partition_index], num_topics=6)
		return lda
		# self.lda_models[partition_index] = lda



	def _check_lda_model(self, partition_index):
		if self.lda_models[partition_index] == None:
			self._train_lda_model(partition_index)
		else:
			return


	def _node_to_bagofneighbors(self,node):
		# currently returns the string of the id's for the neighbors. ***TODO: make __str__ method
		return [str(neighbor) for neighbor in self.b.neighbors(node)]



	def backwards_LDA_psu(self, node):
		'''
			function: backwards_LDA_psu

			params:	node - the node we are recommending for (not the node we want the backwards LDA psu for)

			returns:	scores - a dict mapping nodes to their backwards LDA psu scores

			notes: we want to find argmax_s P(s|u), where u=user s=song. by bayes, argmax_s P(s|u) = argmax_s P(u|s)P(s), the denominator P(u) 
			being cancelled away. we can find P(u|s) by treating the songs as "generating" users (something like backwards LDA). The generative
			story there would be that when a song "generates" a user, i.e when a song is good enough that a user likes it, the way that happens is 
			first the song appeals to some group, and then the song appeals to some person in that group. P(s) can be treated as uniform to
			make recommendations not care about the popularity of the song; else, P(s) could either be estimated by something like playback count,
			or could be found from the graph by degree.
		'''


		# backwards LDA so we take the other partition index (^1); note that "node" is NOT in nodes[partition_index]
		partition_index = (self._find_partition_index(node)^1)
		#just more or less an assertion, should have been trained in init (was needed for old code)
		self._check_lda_model(partition_index)




		scores = {}
		for node_to_check in self.nodes[partition_index]:
			# find P(t|s) as 'gamma'
			bag_rep = self._node_to_bagofneighbors(node_to_check)
			gensim_bow = self.dicts[partition_index].doc2bow(bag_rep)
			gamma, sstats = self.lda_models[partition_index].inference([gensim_bow])
			normalized_gamma = gamma[0]/sum(gamma[0])

			# score is sum over t (P(t|s)P(u|t)) = P(u|s) by independence assumptions of LDA. P(t|s) = normalized_gamma[dist_index], P(u|t) = per_cluster_node_dist[part_index][dist_index][str(node)]. 
			# sum over t:
			score = 0
			for cluster_index in range(len(self.per_cluster_node_distributions[partition_index])):
				#P(t|s):
				p_t_given_s = normalized_gamma[cluster_index]
				# P(u|t):
				p_u_given_t = self.per_cluster_node_distributions[partition_index][dist_index][str(node)]
				# add to score
				score += p_t_given_s * p_u_given_t
				# ____ NOTE: BECAUSE OF 1-NEIGHBOR NODES IN THE GRAPH, NEED TO INCLUDE P(S) PROBABLY ____ ##





			score = sum([self.per_cluster_node_distributions[partition_index][dist_index][str(node)]*normalized_gamma[dist_index] for dist_index in range(len(self.per_cluster_node_distributions[partition_index]))]) #
			scores[node_to_check] = score
		# sort on highest scores
		
		
		
		return scores



	def _find_partition_index(self, node):
		'''
			function: _find_partition_index

			params: node - node to look for

			returns:	partition_index: which partition 'node' is in
		'''


		if node in self.nodes[0]:
			partition_index = 0
		elif node in self.nodes[1]:
			partition_index = 1
		else:
			raise nx.NetworkXException('The node you are trying to recommend for does not exist in the graph')
		return partition_index



	def forwards_LDA_psu(self, node):
		'''
			function: forwards_LDA_psu

			params:	 node - node we are recommending for (not the node we are looking for the score for)

			returns:	scores - dict mapping nodes to their forwards LDA psu scores
						scores_cosine_sim - dict mapping nodes to the cosine similarity between that node's cluster distribution and the original "node"'s cluster distribution

			notes:  again we look for argmax_s P(s|u), but now we model it more directly by running LDA treating users as generating songs. Then argmax
			P(s|u) is simply the most likely song to be generated, which is easily findable by P(t|u)P(s|t). It is unclear to me what the
			relationship between recommendation feature #1 and #2 is... should be somewhat similar. Note that we could be more efficient by sampling
			for example, rather than iterating and picking the best song (sampling like just going and DOING lda's generative story)
			
			for cosine similarity, we simply find the cluster distribution for our node by running inference on our node's "document" (it's neighbors). 
			Then, we find the cluster distribution of an arbitrary other node (which is in the other bipartite set) by running inference on 
			a document consisting only of that node itself. Finally we take the cosine similarity. Note that the inference on a single-word
			document SHOULD be equivalent to iterating through the cluster node distributions and taking that node's probability from each -- not sure which
			is faster.
		'''
		partition_index = self._find_partition_index(node)
		# check to see that the lda model has been trained
		self._check_lda_model(partition_index)

		# get distribution for our node
		normalized_gamma = self._find_forwards_node_cluster_distribution(node, partition_index)
		# bag_rep = self._node_to_bagofneighbors(node)
		# gensim_bow = self.dicts[partition_index].doc2bow(bag_rep)
		# gamma, sstats = self.lda_models[partition_index].inference([gensim_bow])
		# normalized_gamma= gamma[0]/sum(gamma[0])

		scores = {}
		scores_cosine_sim = {}
		for node_to_check in self.nodes[partition_index^1]:
			# cosine similarity:

			# run inference on just the song to find the song's cluster distribution
			new_bag_rep = [str(node_to_check)]
			nnormalized_gamma = self._bag_rep_to_gamma(new_bag_rep, partition_index)
			# new_gensim_bow = self.dicts[partition_index].doc2bow(bag_rep)
			# ngamma, nsstats = self.lda_models[partition_index].inference([gensim_bow])
			# nnormalized_gamma = ngamma[0]/sum(ngamma[0])
			# take cosine similarity
			scores_cosine_sim[node_to_check] = cosine_similarity(nnormalized_gamma, normalized_gamma)

			#probability of generation:


			#score = P(s|u) = sum over t (P(t|u)P(s|t)). P(t|u) = normalized_gamma[dist_index], P(s|t) = per_cluster_node_dist[part_index][dist_index][str(node_to_check)]
			score = sum([self.per_cluster_node_distributions[partition_index][dist_index][str(node_to_check)]*normalized_gamma[dist_index] for dist_index in range(len(self.per_cluster_node_distributions[partition_index]))])
			scores[node_to_check] = score
		
		return scores, scores_cosine_sim

	def _find_backwards_node_cluster_distribution(self, node, partition_index):
		return [distribution[str(node)] for distribution in self.per_cluster_node_distributions[partition_index]]

	def _find_backwards_similarity(self,node, other_node, partition_index):
		'''
			function: _find_backwards_similarity

			params:	node - node we are recommending for
					other_node - node we want to find similarity to
					partition_index - which partition both nodes are in

			returns:	cosine similarity between the two nodes (they are in the same partition)

			notes: backwards similarity uses LDA that treats songs as generating users; each user has a distribution over topics that it is in,
			and backwards similarity is the cosine similarity of that topic distribution vector. Cosine similarity means that users
			who are similar but don't "like" as many songs are still considered similar
		'''

		# other partition index because we're doing backwards LDA
		partition_index = partition_index^1
		# check to see that the lda model has been trained
		self._check_lda_model(partition_index)

		#get the cluster distribution of our node
		node_cluster_distribution = self._find_backwards_node_cluster_distribution(node, partition_index)

		#get the cluster distribution of the other node
		other_node_cluster_distribution = self._find_backwards_node_cluster_distribution(other_node, partition_index)

		#return cosine similarity
		return cosine_similarity(node_cluster_distribution, other_node_cluster_distribution)


	def _bag_rep_to_gamma(self, bag_rep, partition_index):
		gensim_bow = self.dicts[partition_index].doc2bow(bag_rep)
		gamma, sstats = self.lda_models[partition_index].inference([gensim_bow])
		normalized_gamma= gamma[0]/sum(gamma[0])
		return normalized_gamma

	def _find_forwards_node_cluster_distribution(self, node, partition_index):
		bag_rep = self._node_to_bagofneighbors(node)
		return self._bag_rep_to_gamma(bag_rep, partition_index)
		# gensim_bow = self.dicts[partition_index].doc2bow(bag_rep)
		# gamma, sstats = self.lda_models[partition_index].inference([gensim_bow])
		# normalized_gamma= gamma[0]/sum(gamma[0])
		# return normalized_gamma

	def _find_forwards_similarity(self,node, other_node, partition_index):
		'''
			function: _find_forwards_similarity

			params:	node - node we are recommending for
					other_node - node we want to find similarity to
					partition_index - which partition both nodes are in

			returns:	cosine similarity between the two nodes (they are in the same partition)

			notes: forwards similarity models users generating songs. Each user has a per-user cluster distribution.
			we take cosine similarity of these distributions.
		'''

		#check to see that lda model has been trained
		self._check_lda_model(partition_index)

		#get the cluster distribution of our node
		node_cluster_distribution = self._find_forwards_node_cluster_distribution(node, partition_index)

		#get the cluster distribution of the other node
		other_node_cluster_distribution = self._find_forwards_node_cluster_distribution(other_node, partition_index)

		#return cosine similarity
		return cosine_similarity(node_cluster_distribution, other_node_cluster_distribution)


		
	def find_most_similar_user(self,node):
		partition_index = self._find_partition_index(node)
		backwards_sim = {}
		forwards_sim = {}
		# iterate through other nodes on node's side of the bipartite graph
		for other_node in self.nodes[partition_index]:
			backwards_similarity = self._find_backwards_similarity(node, other_node, partition_index)
			forwards_similarity = self._find_forwards_similarity(node, other_node, partition_index)
			# dicts for testing purposes, to see relationships
			backwards_sim[other_node] = backwards_similarity
			forwards_sim[other_node] = forwards_similarity
		return backwards_sim, forwards_sim




	def initialize_feature_mappings(self, node):
		
		backwards_psu = self.backwards_LDA_psu(node)
		
		forwards_psu, forwards_cosine_sim = self.forwards_LDA_psu(node)
		
		backwards_sim, forwards_sim = self.find_most_similar_user(node)

		self.mappings['backwards_psu'] = backwards_psu
		self.mappings['fowards_psu'] = forwards_psu
		self.mappings['forwards_cosine_sim'] = forwards_cosine_sim

		return backwards_psu, forwards_psu, backwards_sim, forwards_sim, forwards_cosine_sim



# ======= [ UTILS ] ========= #
def cosine_similarity(u, v):
	return np.dot(u,v) / (np.sqrt(np.dot(u,u)) * np.sqrt(np.dot(v,v)))

