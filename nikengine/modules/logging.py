from enum import Enum


class loglevel(Enum):
    INFO = 0
    WARN = 1
    ERROR = 2
    FATAL = 3


def printlog(msg, level=loglevel.INFO):
    header = f" "
    new_msg = header + str(msg)
    if level == loglevel.INFO:
        print("[INFO] " + new_msg)
    elif level == loglevel.WARN:
        print("[WARN] " + new_msg)
    elif level == loglevel.ERROR:
        print("[ERROR] " + new_msg)
    elif level == loglevel.FATAL:
        print("[FATAL] " + new_msg)


def pinfo(msg):
    printlog(msg, loglevel.INFO)


def pwarn(msg):
    printlog(msg, loglevel.WARN)


def perr(msg):
    printlog(msg, loglevel.ERROR)


def pfatal(msg):
    printlog(msg, loglevel.FATAL)
