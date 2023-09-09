import csv
import math
import random


def loadCsv(filename):
  lines = csv.reader(open('NBC.csv', 'r'))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
  trainsize = int(len(dataset) * splitRatio)
  trainSet = []
  copy = list(dataset)
  while len(trainSet) < trainsize:
    index = random.randrange(len(copy))
    trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
  seperated = {}
  for i in range(len(dataset)):
    vector = dataset[i]
    if (vector[-1] not in seperated):
      seperated[vector[-1]] = []
    seperated[vector[-1]].append(vector)
  return seperated


def mean(numbers):
  return sum(numbers) / float(len(numbers))


def stdev(numbers):
  avg = mean(numbers)
  variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
  return (math.sqrt(variance))


def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute))
               for attribute in zip(*dataset)]
  del summaries[-1]
  return summaries


def summarize_by_class(dataset):
  seperated = separateByClass(dataset)
  summaries = {}
  for classValue, instances in seperated.items():
    summaries[classValue] = summarize(instances)
  return summaries


def Cal_Prob(x, mean, stdev):
  exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
  return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def Cal_Class_Prob(summeries, inputvector):
  probabilities = {}
  for classValue, classValue, classSummaries in summeries.items():
    probabilities[classValue] = 1
    for i in range(len(classSummaries)):
      mean, stdev = classSummaries[i]
      x = inputvector[i]
      probabilities[classValue] *= Cal_Prob(x, mean, stdev)
    return probabilities

def Predict(summaries,inputVector):
  probs=Cal_Class_Prob(summaries,inputVector)
  bestLabel,bestProb=None,-1
  for classValue,prob in probs.items():
    if bestLabel is None or prob>bestProb:
      bestProb=prob
      bestLabel=classValue
  return bestLabel


def main():
  filename='NBC.csv'
  splitRatio=0.67
  dataset=loadCsv(filename)
  trainingSet,testSet =splitDataset(dataset,splitRatio)
  print('Split {0} rows into train={1} and test={2} rows' .format(len(dataset),len(trainingSet),len(testSet)))
  summaries=summarize_by_class(dataset)
  predictions=Predict(summaries,testSet)
  main()