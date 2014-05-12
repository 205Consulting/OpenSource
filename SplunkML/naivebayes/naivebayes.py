'''
Splunk Naive Bayes class

Ankit Kumar
ankitk@stanford.edu


TODO: Change from P(feature_y=someval) to P(feature_v_somval=True)
'''

import splunklib.client as client
import splunklib.results as results
from collections import defaultdict
import numpy as np
import sys

# use host="localhost",port=8089,username="admin",password="flower00"
# use data_file = "/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes.txt"

class SplunkNaiveBayes(object):

	def __init__(self, host, port, username, password):
		self.service = client.connect(host=host, port=port, username=username, password=password)
		self.jobs = self.service.jobs
		self.mapping = {}
		self.sufficient_statistics = []
		self.class_curr = 0
		self.feature_curr = 0
		self.num_classes = 0
		self.num_features = 0
		
	def update_sufficient_statistics(self,class_val, num_hits, field, value):
		self.sufficient_statistics[self.mapping[class_val]][self.mapping['%s_%s' % (field,value)]] = num_hits
		return




	def sufficient_statistics_splunk_search(self, data_file, feature_fields, class_field):
		csl = self.make_csl(feature_fields + [class_field])
		search_string = 'search source="%s" | table %s | untable %s field value |stats count by %s field value' % (data_file, csl, class_field, class_field)
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




	# def populate_sufficient_statistics(self, data_file, field, class_field):
	# 	search_string = 'search source="%s" | stats count by %s, %s' % (data_file, field, class_field)
	# 	search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	# 	job = self.jobs.create(search_string, **search_kwargs)

	# 	result_count = job["resultCount"]
	# 	offset = 0
	# 	count = 50

	# 	while (offset < int(result_count)):
	# 		kwargs_paginate = {'count':count, 'offset':offset}
	# 		search_results = job.results(**kwargs_paginate)
	# 		for result in results.ResultsReader(search_results):
	# 			class_val = result['%s' % class_field]
	# 			num_hits = int(result['count'])
	# 			field_val = result['%s' % field]
	# 			self.priors[class_val] += num_hits
	# 			self.sufficient_statistics[class_val][field][field_val] += num_hits
	# 		offset += count





	# def initialize_sufficient_statistics(self,data_file,feature_fields, class_field):
	# 	search_string = 'search source="%s" | stats values(%s)' % (data_file, class_field)
	# 	search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	# 	job = self.jobs.create(search_string, **search_kwargs)

	# 	search_results = job.results()
	# 	for result in results.ResultsReader(search_results):
	# 		for class_val in result['values(party)']:
	# 			self.sufficient_statistics[class_val] = {field:defaultdict(int) for field in feature_fields}
	# 		# for feature_field in feature_fields:
	# 		# 	self.sufficient_statistics[]
	# 		# 	for val in result['values(%s)' % feature_field]:
	# 		# 		self.sufficient_statistics[class_val][feature_field][val] = 0
	# 	return




	# 	# sufficient_statistics = {}
	# 	# search_string = 'search source="%s" | stats dc' % data_file
	# 	# print search_string
	# 	# search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	# 	# job = self.jobs.create(search_string, **search_kwargs)

		# search_results = job.results()
		# for result in results.ResultsReader(search_results):
		# 	num_classes = int(result['dc(%s)' % class_field])
		# 	for field in feature_fields:
		# 		sufficient_statistics[field] = np.zeros((num_classes,int(result['dc(%s)' % field])))
				
		# return sufficient_statistics



	def make_csl(self, fields):
		# Makes a comma-seperated list of the fields
		string = ''
		for field in fields[:-1]:
			string += '%s, ' % field
		string += fields[-1]
		return string


	def initialize_sufficient_statistics(self,data_file, feature_fields, class_field):
		'''
			intializes sufficient statistics array by finding out size, and creates the mapping
		'''
		#1: search for all values of all fields in splunk
		search_string = 'search source="%s" | stats values' % (data_file)
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)

		search_results = job.results()
		for result in results.ResultsReader(search_results):
			self.mapping = {result['values(%s)' % class_field][i]:i for i in range(len(result['values(%s)' % class_field]))}
			self.num_classes = len(self.mapping)
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
		


	def train(self, data_file, feature_fields, class_field):
		'''
			trains the classifier

			sufficient statistics are as follows:
			P(class_field=x) for each x
			P(feature_fields[i]=j|class_field=x) for each i,j,x
		'''
		#1: find out how big the sufficient statistic array needs to be, and create the string->index mapping
		self.initialize_sufficient_statistics(data_file, feature_fields, class_field)
		

		#2: create the job that searches for sufficient statistics
		suff_stat_search = self.sufficient_statistics_splunk_search(data_file, feature_fields, class_field)

		#3: populate the sufficient statistics
		self.populate_sufficient_statistics_from_search(suff_stat_search, class_field)

		#4: turn counts into empirical log-probabilities
		self.counts_to_logprobs()
		


	def to_numpy_rep(self, event_to_predict, feature_fields):
		#1: initialize
		np_rep = np.zeros((self.num_features,1))

		#2: add features that the event has; if we've never seen one before, ignore it
		for field in feature_fields:
			val = event_to_predict[field]
			if '%s_%s' % (field, val) in self.mapping:
				np_rep[self.mapping['%s_%s' % (field,val)]] = 1

		#3: return
		return np_rep



	def predict(self, feature_fields, event_to_predict=None):
		if event_to_predict==None:
			event_to_predict = {}
			for feature in feature_fields:
				if np.random.random() > .5:
					event_to_predict[feature] = 'n'
				else:
					event_to_predict[feature] = 'y'


		numpy_rep = self.to_numpy_rep(event_to_predict, feature_fields)
		class_log_prob = np.dot(self.log_prob_suff_stats, numpy_rep)[:,0]
		print class_log_prob
		
		class_log_prob += self.log_prob_priors
		return np.argmax(class_log_prob)




if __name__ == '__main__':
	snb = SplunkNaiveBayes(host="localhost", port=8089, username="admin", password="flower00")
	snb.train("/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes.txt", ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports'],'party')
	print snb.predict(['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports'])