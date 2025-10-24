#!/usr/bin/env python3
import unittest
import sys

if __name__ == '__main__':

    sys.path.append('tests/')
    sys.path.append('src/')
    t_loader = unittest.defaultTestLoader
    t_runner = unittest.TextTestRunner(verbosity=2)
    t = ['test_index', 'test_bm25']

    t_suite = t_loader.loadTestsFromNames(t)
    result = t_runner.run(t_suite)
    sys.exit(not result.wasSuccessful())
