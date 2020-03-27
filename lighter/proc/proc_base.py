import functools
import threading
import logging
import time
from time import sleep
from datetime import datetime, timedelta


# ==== TIMEOUT CONSTANTS ====
# timeout in seconds represents approximately 3 years
TIMEOUT_INFINITY = 99999999
TIMEOUT_24_HOURS = 86400
TIMEOUT_12_HOURS = 43200
TIMEOUT_10_HOURS = 36000
TIMEOUT_5_HOURS = 18000
TIMEOUT_2_HOURS = 7200
TIMEOUT_1_HOUR = 3600
TIMEOUT_30_MINUTES = 1800
TIMEOUT_15_MINUTES = 900
TIMEOUT_10_MINUTES = 600
TIMEOUT_5_MINUTES = 300
TIMEOUT_1_MINUTE = 60
TIMEOUT_30_SECONDS = 30
TIMEOUT_10_SECONDS = 10
TIMEOUT_5_SECONDS = 5
TIMEOUT_1_SECOND = 1


def repeated_run(func, retries=5):
    """
    Decorator for repeating processes. If an exception occurs it can retry the
    last execution.
    :param func: Function on which this decorator is applied. It was mainly meant
    for the run() method of Subprocess.
    :param retries: Maximum number of retries.
    :return: Decorator wrapper.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cnt = 0
        instance = args[0]
        while cnt < retries:
            try:
                func(*args, **kwargs)
                break
            except Exception as e:
                logging.warning('Proc: Process caught exception! Retrying...', e)
                cnt += 1
        if cnt >= retries:
            logging.error('Proc: Failed to complete {} execution...'.format(instance.name))
        instance._done = True
    return wrapper


class Subprocess:
    """
    Base subprocess class for all modules.
    """
    def __init__(self, name, config):
        self.thread = threading.Thread(target=self.run)
        self.config = config
        self.name = name
        self._done = False
        self._success = False
        self.timeout = None
        self.result = None
        logging.debug("Proc: {} created!".format(self.name))

    def success(self):
        return self._success

    def _complete(self):
        self._success = True
        self._done = True
        logging.info("Proc: {} done!".format(self.name))

    def run(self):
        raise NotImplementedError('Subprocess is not defined!')

    def run_async(self):
        if not self.thread.isAlive():
            self.thread.start()
        logging.debug("Proc: {} started async!".format(self.name))

    def join(self, timeout: int = None):
        self.thread.join(timeout=timeout)
        logging.debug("Proc: {} joined!".format(self.name))

    def wait(self, timeout: int = None):
        """
        Attention: Blocking start(). Waits for process to complete or timeout to
        exceed.
        :param timeout: Timeout in seconds. Attention: If not set, then
        wait is set to infinity.
        :return:
        """
        self.run_async()
        # local method caller timeout overwrites global initialized timout if not None
        if timeout is not None:
            self.timeout = timeout
        start_time = datetime.now()
        while not self._done:
            time.sleep(TIMEOUT_1_SECOND)
            if self.timeout is not None and start_time + timedelta(seconds=self.timeout) < datetime.now():
                break
        logging.debug("Proc: {} wait complete after {} sec!".format(self.name, datetime.now() - start_time))
        if self._done:
            logging.info("Proc: Successful {}!".format(self.name))
        else:
            logging.error("Proc: Failed {}!".format(self.name))


class DemoSuccessProcess(Subprocess):
    def __init__(self, name):
        super(DemoSuccessProcess, self).__init__("Demo Success Process: {}".format(name), None)

    @repeated_run
    def run(self):
        logging.info('Proc: Do some heavy computation...')
        sleep(1)
        self._complete()


class DemoFailProcess(Subprocess):
    def __init__(self, name):
        super(DemoFailProcess, self).__init__("Demo Fail Process: {}".format(name), None)
        self.first_run = 0

    @repeated_run
    def run(self):
        if self.first_run == 0:
            logging.info('Proc: Throwing exception.')
            self.first_run += 1
            raise Exception('Demo exception!')
        self._complete()


class DemoFrozenProcess(Subprocess):
    def __init__(self, name):
        super(DemoFrozenProcess, self).__init__("Demo Frozen Process: {}".format(name), None)

    @repeated_run
    def run(self):
        logging.info('Proc: This process gets stuck now... please kill me!')
        while True:
            sleep(0.1)
