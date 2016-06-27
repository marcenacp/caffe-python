import os, sys, subprocess, time, datetime
import numpy as np
from sklearn.metrics import accuracy_score

def train_test_net_command(solver_config_path):
    """
    Train/test process launching cpp solver from shell.
    """
    import caffe
    # Load solver
    solver = None
    solver = caffe.get_solver(solver_config_path)

    # Launch training command
    command = "{caffe} train -solver {solver}".format(caffe=caffe_root + caffe_path,
                                                      solver=solver_config_path)
    subprocess.call(command, shell=True)

def train_test_net_python(solver_path, log_path, accuracy=False,
                          accuracy_metrics=accuracy_score,
                          threshold=0.5, print_every=100,
                          key_label='label', key_score='score',
                          solverstate=None, caffemodel=None,
                          debug=True):
    """
    Pythonic alternative to train/test a network:
    it captures stderr and logs it to a custom log file.

    Errors in C can't be seen in python, so use subprocess.call
    to see if the script terminated.

	solver_path (str)			- Path to the solver's prototxt.
	log_path (str)				- Path to the log directory.
	accuracy (bool)    			- Compute accuracy?
	accuracy_metrics (fun)     	- Accuracy metrics as a function of a net or descriptive string
                                  Possible string accuracies are to be found in caffe.metrics
	threshold (float)			- Threshold
	key_label (str)				- Label key
	key_score (str)				- Score key
    solverstate (string or None)- Full path to the solver state to begin from (weights and training params)
    caffemodel (string or None) - Full path to the Caffe model to begin from (weights only)
	debug (boolean)				- Activate debugging? If True, log won't be captured.
    """
    # Setting log environment variables before importing caffe
    os.environ["GLOG_logtostderr"] = "1"
    if debug:
        os.environ["FLAGS_log_dir"] = log_path
        os.environ["GLOG_log_dir"] = log_path
    import caffe
    from caffe.proto import caffe_pb2
    from pycaffe.utils.output_grabber import *
    from pycaffe.utils.custom_log import *
    from pycaffe.metrics.inference import metrics_from_net
    from pycaffe.metrics.multi_label_metrics import all_measures
	# Start stream redirection using pycaffe.train.output_grabber
    out = start_output(debug, log_path)
    # Get useful parameters from prototxts
    max_iter = get_prototxt_parameter("max_iter", solver_path)
    test_interval = get_prototxt_parameter("test_interval", solver_path)
    display = get_prototxt_parameter("display", solver_path)
    # Get metrics we want to return
    dic = {}
    dic['loss'] = {}
    dic['loss']['train'] = np.empty(max_iter)
    dic['loss']['train'][:] = np.nan
    dic['loss']['test'] = np.empty(max_iter)
    dic['loss']['test'][:] = np.nan
    if accuracy:
        dic['accuracy'] = {}
        dic['accuracy']['test'] = np.empty(max_iter)
        dic['accuracy']['test'][:] = np.nan
    # Work on GPU and load solver
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = None
    solver = caffe.get_solver(solver_path)
    if solverstate is not None:
        solver.restore(solverstate)
    if caffemodel is not None:
        solver.net.copy_from(caffemodel)
    # Log solving
    log_entry(debug, out, "Solving")

    for it in range(max_iter):
        # Iterate
        solver.step(1)

        # Regularly compute metrics on train set
        if it % display == 0:
            dic['loss']['train'][it] = solver.net.blobs['loss'].data

        # Perform adaptive weight noise (regularization method)
        #for _, blob_vec in solver.net.params.iteritems():
        #    for blob in blob_vec:
        #        blob.data[...] += np.random.normal(0, .0075, blob.shape)

        # Regularly compute metrics on test set
        if it % test_interval == 0:
            # DO WE NEED? solver.test_nets[0].forward()
            dic['loss']['test'][it] = solver.test_nets[0].blobs['loss'].data
            if accuracy:
                value_accuracy = metrics_from_net(solver.test_nets[0], accuracy_metrics,
                                                  key_label, key_score, threshold)
                dic['accuracy']['test'][it] = value_accuracy
                log_entry(debug, out, "Test net output #1: accuracy = {}".format(value_accuracy))

    # Break output stream and write to log
    stop_output(out)
    return dic
