from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, hamming_loss

def example_based_measures(y_true, y_pred):
    """
    Evaluation measures used to assess the predictive performance in example-based
    learning: macro/micro precision, macro/micro recall and macro/micro f1
    """
    m = {}
    for cat in ['macro', 'micro']:
        m[cat+'_precision'], m[cat+'_recall'], m[cat+'_f1'], _ = precision_recall_fscore_support(y_true, y_pred, average=cat)
    return m

def label_based_measures(y_true, y_pred):
    """
    Evaluation measures used to assess the predictive performance in multi-label
    label-based learning: hamming_loss, precision, recall and f1
    """
    m = {}
    m['hamming_accuracy'] = 1 - hamming_loss(y_true, y_pred)
    m['precision'], m['recall'], m['f1'], _ = precision_recall_fscore_support(y_true, y_pred)
    return m

def all_measures(y_true, y_pred):
    """
    All measures, both label-based and example-based
    """
    m1 = example_based_measures(y_true, y_pred)
    m2 = label_based_measures(y_true, y_pred)
    return dict(m1.items() + m2.items())
