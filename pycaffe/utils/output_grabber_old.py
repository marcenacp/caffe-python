import sys, os, time, threading
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

    # Log entry to out
    write_output(debug, out, entry)
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


class OutputGrabber(object):
    """
    Class used to grab stderr (default) or another stream
    """
    escape_char = "\b"

    def __init__(self, stream=sys.stderr, threaded=True):
        self.threaded = threaded
        self.origstream = stream
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured
        self.pipeout, self.pipein = os.pipe()
        pass

    def start(self, init=False):
        """
        Start capturing the stream data
        """
        self.capturedtext = ""
        # Save a copy of the stream
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe
        os.dup2(self.pipein, self.origstreamfd)
        if self.threaded:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            time.sleep(0.01)
        # Patch log file with time stamp on first line if initialization
        if init:
            time_stamp = make_time_stamp('%Y/%m/%d %H-%M-%S')
            log_beginning = "Log file created at: {}\n".format(time_stamp)
            self.origstream.write(log_beginning)
        pass

    def write(self, entry):
        self.origstream.write(entry)
        pass

    def stop(self, filename):
        """
        Stop capturing the stream data and save the text in `capturedtext`
        """
        # Flush the stream to make sure all our data goes in before the escape character.
        self.origstream.flush()
        # Print the escape character to make the readOutput method stop
        self.origstream.write(self.escape_char)
        if self.threaded:
            self.workerThread.join()
        else:
            self.readOutput()
        os.close(self.pipeout)
        # Restore the original stream
        os.dup2(self.streamfd, self.origstreamfd)
        #self.origstream.close()
        # Write to file filename
        f = open(filename, "a")
        f.write(self.capturedtext)
        f.close()
		# Correct non glog outputs
        correct_log(filename)
        pass

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`
        """
        while True:
            data = os.read(self.pipeout, 1)  # Read One Byte Only
            if self.escape_char in data:
                break
            if not data:
                break
            self.capturedtext += data
        pass

def start_output(debug, init=False):
    """
    Start stderr grabber
    """
    if not debug:
        out = OutputGrabber()
        out.start(init)
        return out
    return None

def write_output(debug, out, entry):
    if not debug:
        out.write(entry)
    pass

def purge_output(debug, out, log_path):
    """
    Stop and start stderr grabber in the same log file
    """
    if not debug:
        stop_output(debug, out, log_path)
        new_out = start_output(debug)
        return new_out
    return None

def stop_output(debug, out, log_path):
    """
    Stop stderr grabber and close files
    """
    if not debug:
        out.stop(log_path)
    pass
