'''
SplunkML classifiers

Ankit Kumar
ankitk@stanford.edu


Todo: 	(1) get better with numpy arrays...
		(2) add laplace smoothing
'''

import splunklib.client as client
import splunklib.results as results
from sklearn.naive_bayes import BernoulliNB
from collections import defaultdict
import numpy as np
import sys

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes_correct.txt"'
vote_class = 'party'

# use host="localhost",port=8089,username="admin",password="flower00"
# use data_file = "/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes.txt"

class SplunkClassifierBase(object):
	''' SplunkClassifierBase

	Base class of splunk classifiers. Functionality includes evaluate_accuracy(feature_fields, class_field)

	'''


	def __init__(self, host, port, username, password):
		self.service = client.connect(host=host, port=port, username=username, password=password)
		self.jobs = self.service.jobs
		self.trained = False
		self.feature_fields = None


	def predict(self, feature_fields, class_field, event_to_predict):
		'''
			to overwrite
		'''
		pass


	def train_classifier(self, search_string, feature_fields, class_field):
		'''
			to overwrite
		'''
		pass

	def train(self, search_string, feature_fields, class_field):
		self.train_classifier(search_string, feature_fields, class_field)
		self.trained=True
		

	def compare_sklearn(self, np_reps, gold):
		'''
			to overwrite
		'''
		pass


	def check_accuracy(self, search_string, feature_fields, class_field):
		'''
			check_accuracy(search_string, feature_fields, class_field)

			search_string: string to use in the splunk search to narrow events
			feature_fields: which fields to use to predict
			class_field: field to predict

			returns: accuracy of prediction

			notes: assumes that classifier is already trained. calls predict on each event.
		'''
		# 1: check that classifier is trained:
		if not self.trained:
			raise 'classifier is not trained'

		# 2: search for the events
		search_string = 'search %s | eventstats values' % search_string
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)
		result_count = int(job["resultCount"])
		
		# 3: iterate through events, calling predict on each one. record results.
		correct = 0
		offset = 0
		count = 50
		np_reps = []
		gold = []
		while (offset < result_count):
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):
				predicted_class, np_rep, actual_class = self.predict(feature_fields, class_field,result, return_numpy_rep=True)
				np_reps.append(np_rep)
				gold.append(actual_class)
				if predicted_class == result[class_field]:
					correct += 1
			offset += count

		# 4: calculate percentage
		perc_correct = float(correct)/result_count

		# 5: check sklearn's implementation
		sklearn_accuracy = self.compare_sklearn(np_reps, gold)

		# 5: return
		return perc_correct, sklearn_accuracy


	def evaluate_accuracy(self, search_string, feature_fields, class_field):
		'''
			evaluate_accuracy()

			trains the classifier, then predicts each of the events it trains on and records how many were correct
		'''
		print "Now evaluating %s test set accuracy." % self.__class__.__name__

		#1 : train the classifier
		print "--> Training the classifier..."
		self.train(search_string, feature_fields, class_field)
		print "... done."

		#2 : check accuracy
		print "--> Iterating through test set and checking accuracy, then comparing with sklearn..."
		accuracy, sklearn_accuracy = self.check_accuracy(search_string, feature_fields, class_field)
		print "done."

		#3 : return
		print "Accuracy was %f. Sklearn's was %f." % (accuracy, sklearn_accuracy)

		return accuracy






class SplunkNaiveBayes(SplunkClassifierBase):

	def __init__(self, host, port, username, password):
		super(SplunkNaiveBayes, self).__init__(host, port, username, password)
		self.mapping = {}
		self.sufficient_statistics = []
		self.class_curr = 0
		self.feature_curr = 0
		self.num_classes = 0
		self.num_features = 0
		
	def update_sufficient_statistics(self,class_val, num_hits, field, value):
		self.sufficient_statistics[self.mapping[class_val]][self.mapping['%s_%s' % (field,value)]] = num_hits
		return




	def sufficient_statistics_splunk_search(self, search_string, feature_fields, class_field):
		csl = self.make_csl(feature_fields + [class_field])
		search_string = 'search %s | table %s | untable %s field value |stats count by %s field value' % (search_string, csl, class_field, class_field)
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)
		return job


	def populate_sufficient_statistics_from_search(self, job, class_field):
		result_count = job["resultCount"]
		offset = 0
		count = 50

		while (offset < int(result_count)):
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):
					class_val = result['%s' % class_field]
					num_hits = int(result['count'])
					field = result['field']
					value = result['value']
					self.update_sufficient_statistics(class_val, num_hits, field, value)
			offset += count



	def make_csl(self, fields):
		# Makes a comma-seperated list of the fields
		string = ''
		for field in fields[:-1]:
			string += '%s, ' % field
		string += fields[-1]
		return string


	def initialize_sufficient_statistics(self,search_string, feature_fields, class_field):
		'''
			intializes sufficient statistics array by finding out size, and creates the mapping
		'''
		#1: search for all values of all fields in splunk
		search_string = 'search %s | stats values' % (search_string)
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)

		search_results = job.results()
		for result in results.ResultsReader(search_results):
			self.mapping = {result['values(%s)' % class_field][i]:i for i in range(len(result['values(%s)' % class_field]))}
			self.num_classes = len(self.mapping)
			for elem in self.mapping.items():
				self.mapping[elem[1]] = elem[0]
			curr_index = 0
			for field in feature_fields:
				for value in result['values(%s)' % field]:
					self.mapping['%s_%s' % (field,value)] = curr_index
					curr_index += 1

		self.num_features = curr_index

		#2: make the numpy array for the sufficient statistics:
		self.sufficient_statistics = np.zeros((self.num_classes,self.num_features))

		return



	def counts_to_logprobs(self):
		#1: sufficient stat log probabilities
		probabs = self.sufficient_statistics / self.sufficient_statistics.sum(axis=1)[:,np.newaxis]
		self.log_prob_suff_stats = np.log(probabs)

		#2: priors
		'''
			note: will this always work? what about if soem event is "missing" a field, will this not work (i.e should I also do a splunk search to find priors once?)
			perhaps close enough.
		'''

		priors = self.sufficient_statistics.sum(axis=1)
		priors = priors / priors.sum()
		self.log_prob_priors = np.log(priors)
		


	def train_classifier(self, search_string, feature_fields, class_field):
		'''
			train_classifier(search_string, feature_fields, class_field)

			search_string: string to search splunk with
			feature_fields: fields to use as features
			class_field: field to predict

			returns: nothing, but sufficient statistics are populated

			notes: sufficient statistics are priors (P(c=x)) for each class c, and P(x_i=true) for each x_i, where an x_i exists for each field-value pair in the feature fields
		'''
		#1: find out how big the sufficient statistic array needs to be, and create the string->index mapping
		self.initialize_sufficient_statistics(search_string, feature_fields, class_field)
		

		#2: create the job that searches for sufficient statistics
		suff_stat_search = self.sufficient_statistics_splunk_search(search_string, feature_fields, class_field)

		#3: populate the sufficient statistics
		self.populate_sufficient_statistics_from_search(suff_stat_search, class_field)

		#4: turn counts into empirical log-probabilities
		self.counts_to_logprobs()
		


	def to_numpy_rep(self, event_to_predict, feature_fields):
		#1: initialize
		np_rep = np.zeros((self.num_features,1))

		#2: add features that the event has; if we've never seen one before, ignore it
		for field in feature_fields:
			if field not in event_to_predict:
				continue
			val = event_to_predict[field]
			if '%s_%s' % (field, val) in self.mapping:
				np_rep[self.mapping['%s_%s' % (field,val)]] = 1

		#3: return
		return np_rep



	def predict(self, feature_fields, class_field, event_to_predict, return_numpy_rep=False):
		'''
			predict(*)

			notes: uses naive bayes assumption: P(c=x) is proportional P(x_i's|c)P(c). P(c) is the prior, P(x_i's|c) decomposes to 
			P(x_1|c)P(x_2|c)...P(x_n|c); these are all calculated in log space using dot product.
		'''
		numpy_rep = self.to_numpy_rep(event_to_predict, feature_fields)
		class_log_prob = np.dot(self.log_prob_suff_stats, numpy_rep)[:,0]
		class_log_prob += self.log_prob_priors
		if return_numpy_rep:
			actual_class = self.mapping[event_to_predict[class_field]]
			return self.mapping[np.argmax(class_log_prob)], numpy_rep.T[0], actual_class
		else:
			return self.mapping[np.argmax(class_log_prob)]


	def compare_sklearn(self, np_reps, gold):
		X = np.array(np_reps)
		nb = BernoulliNB(alpha=0)
		y = np.array(gold)
		nb.fit(X,y)
		return nb.score(X,y)





if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	snb = SplunkNaiveBayes(host="localhost", port=8089, username=username, password=password)
	snb.evaluate_accuracy(vote_search, vote_features, vote_class)







