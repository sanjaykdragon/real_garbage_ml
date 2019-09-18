import numpy as np
import random
from sklearn.linear_model import LinearRegression

TRAIN_SET_COUNT = 1000
TRAIN_INPUT = list()
TRAIN_OUTPUT = list()

def fill_examples():
  for i in range(TRAIN_SET_COUNT):
    a = random.uniform(0, 10)
    op = a * random.uniform(20000, 50000)
    TRAIN_INPUT.append([a])
    TRAIN_OUTPUT.append(op)
  print("filled examples")


def do_learn_and_pred():
  fill_examples();
  predictor = LinearRegression(n_jobs=-1).fit(TRAIN_INPUT, TRAIN_OUTPUT) 
  #create logistic_regression and fit it  

  X_TEST = [[3.7], [4.2], [5.7]]
  outcome = predictor.predict(X_TEST) #predict
  score = predictor.score(X_TEST, outcome)
  #coefficients = predictor.coef_

  print(outcome)
  print("===============================");
  print(score)

do_learn_and_pred()
