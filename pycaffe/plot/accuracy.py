import matplotlib, os
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from nideep.eval.learning_curve import LearningCurve
from nideep.eval.eval_utils import Phase
import nideep.eval.log_utils as lu

from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams.update({'font.size': 15})

def print_learning_curve(net_prefix, log_path, fig_path, accuracy=True, format_x_axis=True):
    """
    Print learning curve inline (for jupiter notebook) or by saving it.
    """
    fig_path = os.path.abspath(fig_path)
    e = LearningCurve(log_path)
    e.parse()

    for phase in [Phase.TRAIN, Phase.TEST]:
        num_iter = e.list('NumIters', phase)
        loss = e.list('loss', phase)
        plt.plot(num_iter, loss, label='on %s set' % (phase.lower(),))

        plt.xlabel('iteration')
        if format_x_axis:
            ticks, _ = plt.xticks()
            plt.xticks(ticks, ["%dK" % int(t/1000) for t in ticks])
    plt.ylabel('loss')
    plt.title(net_prefix +' on train and test sets')
    plt.legend()
    plt.grid()
    plt.savefig(fig_path + "/" + net_prefix + "_learning_curve.png")

    if accuracy:
        plt.figure()
        num_iter = e.list('NumIters', phase)
        acc = e.list('accuracy', phase)
        plt.plot(num_iter, acc, label=e.name())

        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.title(net_prefix+" on %s set" % (phase.lower(),))
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(fig_path + "/" + net_prefix + "_accuracy.png")

def print_learning_curve_from_dic(dic, net_prefix, fig_path):
    """
    Print learning curve using the dictionary
    as returned by pycaffe.train.train_test.
    """
    for metric in dic.keys():
        for phase in dic[metric].keys():
            metric_value = dic[metric][phase]
            mask = np.isfinite(metric_value)
            num_iter = np.arange(len(metric_value))
            plt.plot(num_iter[mask], metric_value[mask], label='on %s set' % phase)
            plt.xlabel('iteration')
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.grid()
        plt.savefig(fig_path + "/" + net_prefix + "_" + metric + ".png")
        plt.clf()
