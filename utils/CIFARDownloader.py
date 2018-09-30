import glob
import math
import os
import tarfile

import requests
import tqdm

try:
    from utils.FormattedLogger import FormattedLogger
except ImportError:
    import sys

    project_path = os.path.abspath(os.getcwd())
    print("Appending {} to sys.path".format(project_path))
    sys.path.append(project_path)
    from utils.FormattedLogger import FormattedLogger

import utils.filesystem


class CIFARDownloaderException(Exception):
    pass


class CIFARDownloader(object):
    """CIFAR dataset downloader. """
    cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-{}-python.tar.gz'

    def __init__(self, cifar_size=10, local_dir='data'):
        """
        Initialise the downloader.

        :param int cifar_size:
        :param str local_dir:
        """
        # Verify input
        if cifar_size in [10, 100]:
            self.cifar_size = cifar_size
        else:
            raise CIFARDownloaderException("CIFAR dataset type has to be 10 or 100, got {}".format(cifar_size))

        self.download_url = self.cifar_url.format(self.cifar_size)
        utils.filesystem.make_dir(local_dir)
        self.local_dir = local_dir
        self.archive_path = os.path.join(self.local_dir, self.download_url.split('/')[-1])

        self.logger = FormattedLogger('CIFARDownloader')
        self.logger.info("Initialised a CIFAR-{} downloader.".format(self.cifar_size))

    def __str__(self):
        return "CIFAR{}Downloader".format(self.cifar_size)

    def _download(self):
        try:
            stream_handle = requests.get(self.download_url, stream=True)
            total_size = int(stream_handle.headers.get('content-length', 0))
            block_size = 1024
            kb_written = 0
            with open(self.archive_path, 'wb') as archive_handle:
                self.logger.info("Downloading {} to {}".format(self.download_url, self.archive_path))
                for data_chunk in tqdm.tqdm(stream_handle.iter_content(block_size),
                                            total=math.ceil(total_size // block_size),
                                            unit='kB', unit_scale=False):
                    kb_written = kb_written + len(data_chunk)
                    archive_handle.write(data_chunk)
            if total_size != 0 and kb_written != total_size:
                self.logger.error("Download failed.")
        except Exception as ex:
            raise CIFARDownloaderException("Error: {}".format(ex))

    def _extract(self, retry_download=True):
        self.logger.info("Extracting {} to {}".format(self.archive_path, self.local_dir))
        try:
            cifar_tar = tarfile.open(self.archive_path, 'r:gz')
            cifar_tar.extractall(self.local_dir)
            cifar_tar.close()
        except (IOError, EOFError, tarfile.ReadError):
            if retry_download:
                self.logger.error(
                    "Present archive {} is incomplete: removing the archive and attempting to re-download.".format(
                        self.archive_path))
                utils.filesystem.remove_all(self.archive_path)
                self._download()
                self._extract(retry_download=False)
            else:
                raise CIFARDownloaderException("Corrupt archive, re-download failed -- giving up.")
        else:
            self.logger.info("Removing archive: {}".format(self.archive_path))
            utils.filesystem.remove_all(self.archive_path)

    def get_data(self):
        data_dir = os.path.join(self.local_dir, 'cifar-{}-*'.format(self.cifar_size))
        try:
            data_dir = glob.glob(data_dir)[0]
        except IndexError:
            raise CIFARDownloaderException("No paths matching {}, quitting.".format(data_dir))

        def _data_exists_msg():
            self.logger.info("All CIFAR-{} data is present and accounted for.".format(self.cifar_size))

        def _check_cifar10():
            return True if len(glob.glob(os.path.join(data_dir, 'data*'))) == 5 else False

        def _check_cifar100():
            return True if len(
                set([os.path.basename(fname) for fname in glob.glob(os.path.join(data_dir, '*'))]).intersection(
                    {'meta', 'test', 'train'})) == 3 else False

        self.logger.info("Checking if CIFAR-{} data is present...".format(self.cifar_size))
        if self.cifar_size == 10:
            if _check_cifar10():
                _data_exists_msg()
                return data_dir
        elif self.cifar_size == 100:
            if _check_cifar100():
                _data_exists_msg()
                return data_dir
        self.logger.warning("Data is not (all) present, attempting to download.")
        if not os.path.exists(self.archive_path):
            self._download()
        else:
            self.logger.info("{} already exists, skipping the download.".format(self.archive_path))

        if not os.path.exists(self.archive_path):
            raise CIFARDownloaderException(
                "Error: archive {} does not exist. Check if the download has succeeded.".format(self.archive_path))
        self._extract()
        data_present = _check_cifar10() if self.cifar_size == 10 else _check_cifar100()
        if not data_present:
            raise CIFARDownloaderException("Error: downloaded data did not pass integrity checks.")
        else:
            _data_exists_msg()

        return data_dir


if __name__ == '__main__':
    cifar_downloader = CIFARDownloader(cifar_size=10)
    cifar_downloader.get_data()
