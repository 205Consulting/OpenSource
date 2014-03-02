



class mytest(object):
	def __init__(self):
		self.x = 33



	def realtest(self, index):
		return index + self.x

	def tester(self):
		x = map(self.realtest, range(10))
		print x