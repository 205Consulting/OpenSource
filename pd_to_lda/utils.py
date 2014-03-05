from functools import partial



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