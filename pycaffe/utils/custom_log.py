import time, datetime

def make_time_stamp(pattern):
    """
    Time stamp with explicit string pattern.
	A possible pattern would be '%Y%m%d_%H%M%S'.
    """
    now = time.time()
    return datetime.datetime.fromtimestamp(now).strftime(pattern)

def get_prototxt_parameter(param, prototxt):
    with open(prototxt) as f:
        try:
            line = [s for s in f.readlines() if param in s][-1]
            param_value = int(line.rstrip().split(": ")[1].split("#")[0])
        except IndexError:
            print "{} not defined in prototxt {} or bad layout.".format(param, prototxt)
    return param_value
