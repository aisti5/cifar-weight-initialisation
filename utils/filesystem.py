import contextlib
import logging
import os
import shutil
import tempfile

from utils.FormattedLogger import FormattedLogger

logger = FormattedLogger('CIFARUtils')


@contextlib.contextmanager
def temp_dir_context():
    """Create a temporary folder which get deleted outside the context."""
    tmp_dir = tempfile.mkdtemp()
    logger.debug("Created temp folder at: {}".format(tmp_dir))
    try:
        yield tmp_dir
    finally:
        remove_all(tmp_dir)


def remove_all(path, ignore_errors=True):
    """
    Remove path.

    :param str path:            Path to be removed.
    :param bool ignore_errors:  If False, function will raise on errors.
    """
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=ignore_errors)
        else:
            os.unlink(path)
        logger.debug("Removed {}".format(path))
    except (OSError, IOError) as ex:
        if ignore_errors:
            logger.error("Error removing {}: {}".format(path, ex))
        else:
            raise


def make_dir(path):
    """
    Created nested directories.

    :param str path:     Path to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.debug("Created folder {}".format(path))
    else:
        logger.debug("{} already exists".format(path))
