from functools import partial
from pd_lda import pd_lda
import pandas as pd

df = pd.read_pickle('activities.df')
df_one = df[:len(df)-20]
df_two = df[len(df)-20:len(df)]
pdlda = pd_lda()
model = pdlda.update_lda(df_one, ['words'])
print model.show_topic(1)
# mymodel = model.copy()
new_model = pdlda.update_lda(df_two, ['words'])
print new_model.show_topic(1)






class mytest(object):
	def __init__(self):
		self.x = 33



	def realtest(self, index):
		return index + self.x

	def tester(self):
		x = map(self.realtest, range(10))
		print x

	def func1(self, y, z):
		print "first: " + str(y)
		print "second: " + str(z)

	def test(self):
		new = partial(self.func1, 2)
		new(3)