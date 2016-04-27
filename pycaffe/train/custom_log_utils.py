def make_time_stamp(pattern):
    """
    Time stamp with explicit string pattern.
	A possible pattern would be '%Y%m%d_%H%M%S'.
    """
    now = time.time()
    return datetime.datetime.fromtimestamp(now).strftime(pattern)

def log_entry(debug, out, text):
    """
    Standardized log entries for glog
    """
    # Format log entry
    monthday = make_time_stamp('%m%d')
    time_stamp = make_time_stamp('%H:%M:%S')
    now = time.time()
    ms = "."+str('%06d' % int((now - int(now)) * 1000000))
    line_form = "I{monthday} {time_stamp}  0000 main.py:00] {text}\n"
    entry = line_form.format(monthday=monthday, time_stamp=time_stamp+ms, text=text)

    # Log entry to out
    write_output(debug, out, entry)
    pass

def get_prototxt_parameter(param, prototxt):
    with open(prototxt) as f:
        try:
            line = [s for s in f.readlines() if param in s][-1]
            param_value = int(line.rstrip().split(": ")[1].split("#")[0])
        except IndexError:
            print "{} not defined in prototxt {} or bad layout.".format(param, prototxt)
    return param_value
