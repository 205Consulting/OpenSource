import pandas as pd
import genism
import operator
import sys,os,io



class pd_lda(object):

	def __init__(self, df, fields):
		'''
			function: init (df, fields)

			params: df - dataframe with entries to run lda on
					

			returns: instantiated pd_lda class
		'''
		self.df = df
		
		


	def add_lda_column(self, fields):
		'''
			function: add_lda_column

			params: fields - list of fields to concatenate and run lda on

			returns: the original dataframe with a new column corresponding the the lda inference vector given by the lda model trained on
			the fields passed in. Note that self.df will be updated as well.
		'''
		# 1: make LDA model
		LDA_model = self.build_lda_model(fields)

		# 2: update dataframe
		self.df['LDA_%s' % "".join(fields)] = self.df["".join(fields)].map(lambda x: LDA_model.inference(LDA_model.inference([LDA_model.id2word.doc2bow(x)])[0][0]))

		# 3: return dataframe
		return self.df





	def generate_corpus(self, fields):
		'''
			function: generate_corpus

			params: fields - list of fields to concatenate

			returns: gensim style corpus given by self.df and the fields given
		'''


		# 1: make a list of lists, to be filled by each row in self.df
		texts = [[] for i in range(len(self.df))]

		# 2: for each field, concatenate each text with the fields contents
		for field in fields:
			field_values = list(self.df[field])
			texts = map(operator.add, texts, field_values)


		# 3: add the resulting texts as a new field ino self.df
		field_string = "".join(fields)
		self.df[field_string] = pd.Series(texts)

		# 4: convert to gensim style objects and return
		dictionary = gensim.corpora.Dictionary(all_texts)
		corpus = [dictionary.doc2bow(text) for text in all_texts]
		return corpus, dictionary





	def build_lda_model(self, fields):
		'''
			function: build_lda_model(df, fields)

			params: fields - list of fields corresponding to columns in the df to run lda on

			returns: gensim LDAModel class trained on the data given
		'''
		# 1: make corpus and dictionary
		corpus,dictionary = self.generate_corpus(fields)

		# 2: run lda and return
		lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
		return lda, dictionary


