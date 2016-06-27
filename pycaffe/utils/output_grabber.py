import sys, os, time
from pycaffe.utils.custom_log import make_time_stamp

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

    # Log entry to stderr
    sys.stderr.write(entry)
    pass

def correct_log(log_path):
    """
    Remove non-glog lines
    """
    with open(log_path, "r") as f:
        lines = f.readlines()
    with open(log_path, "w") as f:
        for line in lines:
            if line[0] == "I" or line[0:19] == "Log file created at":
                f.write(line)
    pass

def list_fds():
    """
    List of all open fds for debugging
    """
    for num in os.listdir('/proc/self/fd/'):
        try:
            path = os.readlink(os.path.join('/proc/self/fd', num))
            print(num, path)
        except: # TODO remove FileNotFoundError:
            print(num, '[closed]')


class OutputGrabber(object):
    """
    Class used to grab stderr (default) or another stream
    """
    def __init__(self, stream=sys.stderr, output_path="/tmp"):
        self.replaced_stream = stream
        if os.path.isdir(output_path):
            self.output_path = os.path.abspath(output_path) + "/caffe.INFO"
        else:
            raise ValueError("Please, provide a valid output path for logging")

    def start(self):
        self.replaced_fd = self.replaced_stream.fileno()
        self.backup_fd = os.dup(self.replaced_stream.fileno())
        self.output_file = open(self.output_path, 'a')
        self.replaced_stream.flush()
        os.dup2(self.output_file.fileno(), self.replaced_stream.fileno())

    def stop(self):
        self.output_file.flush()
        os.dup2(self.backup_fd, self.replaced_fd)
        self.output_file.close()
        os.close(self.backup_fd)

    def write(self, entry):
        self.replaced_stream.write(entry)
        pass

def start_output(debug, output_path):
    """
    Start stderr grabber
    """
    if not debug:
        out = OutputGrabber(output_path=output_path)
        out.start()
        return out
    return None

def write_output(out, entry):
    if out is not None:
        out.write(entry)
    pass

def purge_output(out):
    """
    Stop and start stderr grabber in the same log file
    """
    debug = out is None
    if not debug:
        output_path = out.output_path
        stop_output(out)
        new_out = start_output(debug=debug, output_path=output_path)
        return new_out
    return None

def stop_output(out):
    """
    Stop stderr grabber and close files
    """
    if out is not None:
        out.stop()
    pass
