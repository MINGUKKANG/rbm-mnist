# -*- coding: utf-8 -*-

from load_data import load_data

def test():
  print '... loading data ...'

  datasets = load_data('mnist.pkl.gz')

  train_set_x, train_set_y = datasets[0]
  test_set_x, test_set_y = datasets[2]


if __name__ == '__main__':
  test()