import os
import sys
import subprocess
import time, datetime

# Setting log environment variables before importing caffe
use_python = True
os.environ["GLOG_minloglevel"] = "0"
if use_python:
    os.environ["GLOG_logtostderr"] = "1" # only if using custom log
else:
    os.environ["GLOG_log_dir"] = log_path # only if using glog

# Python and caffe
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from pycaffe.utils.output_grabber import *
from pycaffe.utils.custom_log import *
from pycaffe.metrics.inference import metrics_from_net
from pycaffe.metrics.multi_label_metrics import all_measures
from sklearn.metrics import accuracy_score

def train_test_net_command(solver_config_path):
    """
    Train/test process launching cpp solver from shell.
    """
    # Load solver
    solver = None
    solver = caffe.get_solver(solver_config_path)

    # Launch training command
    command = "{caffe} train -solver {solver}".format(caffe=caffe_root + caffe_path,
                                                      solver=solver_config_path)
    subprocess.call(command, shell=True)

def train_test_net_python(solver_path, log_path, accuracy=False, accuracy_metrics=accuracy_score, key_label='label', key_score='score', threshold=0.5, print_every=100, debug=False):
    """
    Pythonic alternative to train/test a network:
    it captures stderr and logs it to a custom log file.

    Errors in C can't be seen in python, so use subprocess.call
    to see if the script terminated.

	solver_path (str)			- Path to the solver's prototxt.
	log_path (str)				- Path to the log file.
	accuracy (boolean)			- Compute accuracy?
	accuracy_metrics (function)	- Accuracy metrics as a function of a net or descriptive string
                                  Possible string accuracies are to be found in caffe.metrics:
                                  ['macro_precision', 'micro_precision', 'macro_recall',
                                   'micro_recall', 'macro_f1', 'micro_f1', 'precision',
                                   'recall', 'f1', 'hamming_accuracy']
	key_label (str)				-
	key_score (str)				-
	threshold (float)			-
	print_every (int)			- Prints indications to the command line every print_every iteration.
	debug (boolean)				- Activate debugging? If True, log won't be captured.
    """
	# Start stream redirection using pycaffe.train.output_grabber
    start_time = time.time()
    out = start_output(debug, init=True)
    # Get useful parameters from prototxts
    max_iter = get_prototxt_parameter("max_iter", solver_path)
    test_interval = get_prototxt_parameter("test_interval", solver_path)
    # Work on GPU and load solver
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = None
    solver = caffe.get_solver(solver_path)
    # Log solving
    log_entry(debug, out, "Solving")
    # Selected accuracy metrics
    possible_values = ['macro_precision', 'micro_precision', 'macro_recall', 'micro_recall', 'macro_f1', 'micro_f1', 'precision', 'recall', 'f1', 'hamming_accuracy']
    if isinstance(accuracy_metrics, basestring):
        key_accuracy = accuracy_metrics
        try:
            possible_values.index(key_accuracy)
            accuracy_metrics = lambda y_true, y_pred: all_measures(y_true, y_pred)[key_accuracy]
        except ValueError:
            print "{} is not a valid metrics, please choose between possible values: ".format(key_accuracy, possible_values)

    for it in range(max_iter):
        # Iterate
        solver.step(1)

        # Regularly compute accuracy on test set
        if accuracy:
            if it % test_interval == 0:
                value_accuracy = metrics_from_net(solver.test_nets[0], accuracy_metrics, key_label, key_score, threshold)
                log_entry(debug, out, "Test net output #1: accuracy = {}".format(value_accuracy))

        # Regularly print iteration
        if it % print_every == 0:
            print "Iteration", it

        # Regularly purge stderr/output grabber
        if it % 1000 == 0:
            out = purge_output(debug, out, log_path)
    # Break output stream and write to log
    stop_output(debug, out, log_path)
    pass
