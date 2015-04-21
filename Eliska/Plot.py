import Cache
import matplotlib.pyplot as plt
import argparse

def main():
	parser = argparse.ArgumentParser(description='Plot log likelihood')
	parser.add_argument('-l', '--likelihood', type=str, help='Likelihood cache file', required=True)
	parser.add_argument('-t', '--title', default='', type=str, help='Plot title', required=False)
	args = vars(parser.parse_args())
	plot(readValues(args['likelihood']), args['title'])

def readValues(fileName):
	print fileName
	likelihood = Cache.Cache(fileName, [])
	return likelihood.cache

def plot(values, title):
	print values
	plt.plot(values)
	plt.title(title)
	plt.ylabel('Log likelihood')
	plt.xlabel('Iteration')
	plt.xticks(range(1, len(values)+1))
	plt.show()

if __name__ == '__main__':
    main()