# -*- coding: utf-8 -*-

from contextlib import contextmanager
from datetime import timedelta
import signal
import slack

__all__ = [
    "debug",
    "time_limit",
    "print_progress_bar",
    "slack_message",
]

debug = False


@contextmanager
def time_limit(seconds):
    """
    ex)
    timeout = 5
    try:
        with time_limit(timeout):
            some_work()
    except TimeoutException:
        handle_TimeoutException()
    """

    def signal_handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


def print_progress_bar(
    iteration,
    total,
    prefix,
    start_time,
    current_time,
    decimals=1,
    length=20,
    fill="█",
    printEnd="\r",
):
    """
    progress bar for iteration
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    suffix = str(timedelta(seconds=int(current_time - start_time)))
    message = prefix + " |" + bar + "| " + percent + "% " + suffix
    print("\r" + message, end=printEnd)

    # Print New Line on Complete
    if iteration == total:
        print()

    return message


slack_client = slack.WebClient(
    token="xoxp-880429464020-868103546978-870635433953-b3c779132c2eeb887cb6971d7091836f"
)


def slack_message(message, channel="debug"):
    """
    Send message to slack workspace (currently, jyuno426.slack.com)
    """

    if debug:
        channel = "debug"

    slack_client.chat_postMessage(channel=channel, text=message)
