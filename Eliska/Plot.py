import Cache
import matplotlib.pyplot as plt
import argparse

def main():
	parser = argparse.ArgumentParser(description='Plot log likelihood')
	parser.add_argument('-c', '--cache', type=str, help='Likelihood cache file', required=True)
	parser.add_argument('-t', '--title', default='', type=str, help='Plot title', required=False)
	parser.add_argument('-l', '--limit', default=0, type=int, help='Iterations limit', required=False)
	args = vars(parser.parse_args())
	
	plot(readValues(args['cache'], args['limit']), args['title'])

def readValues(fileName, limit=0):
	print fileName
	likelihood = Cache.Cache(fileName, [])
	if limit:
		likelihood.cache = likelihood.cache[:limit]
	return likelihood.cache

def plot(values, title):
	print values
	plt.plot(range(1,len(values)+1), values)
	plt.title(title)
	plt.ylabel('Log likelihood')
	plt.xlabel('Iteration')
	plt.xticks(range(1, len(values)+1))
	plt.gca().set_xlim(left=1)
	plt.gca().set_xlim(right=len(values))
	plt.show()

if __name__ == '__main__':
    main()