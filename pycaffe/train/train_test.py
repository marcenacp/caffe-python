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

# python and caffe
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from pycaffe.train.output_grabber import *
from pycaffe.train.custom_log_utils import *

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

def train_test_net_python(solver_path, log_path, accuracy=False, print_every=100, debug=False):
    """
    Pythonic alternative to train/test a network:
    it captures stderr and logs it to a custom log file.

    Errors in C can't be seen in python, so use subprocess.call
    to see if the script terminated.

	solver_path		- str - Path to the solver's prototxt.
	log_path		- str - Path to the log file.
	accuracy		- boolean - Compute accuracy?
	print_every		- int - Prints indications to the command line every print_every iteration.
	debug			- boolean - Activate debugging? If True, log won't be captured.
    """
    from sklearn.metrics import recall_score, accuracy_score
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

    for it in range(max_iter):
        # Iterate
        solver.step(1)

        # Regularly compute accuracy on test set
        if accuracy:
            if it % test_interval == 0:
                solver.test_nets[0].forward()
                # retrieve labels and predictions
                y_true = solver.test_nets[0].blobs['label'].data
                y_prob = solver.test_nets[0].blobs['score'].data
                # reshape labels and predictions
                y_true = np.squeeze(y_true)
                y_prob = np.squeeze(y_prob)
                if y_true.ndim == 1:
                    n = y_true.shape[0]
                    y_true.reshape(1, n)
                    y_prob.reshape(1, n)
                y_pred = np.array([[prob>=0.5 for prob in preds] for preds in y_prob])
                value_accuracy = accuracy_score(y_true, y_pred)
                #value_accuracy = recall_score(y_true, y_pred, average='macro')
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
