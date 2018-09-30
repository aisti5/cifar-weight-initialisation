import os
import unittest
import unittest.mock as mock

import utils.CIFARDownloader
from utils.filesystem import temp_dir_context


class TestDownloader(unittest.TestCase):
    """Tests related to CIFARDownloader functionality."""
    cifar_archive = 'cifar-{}-python.tar.gz'

    def test_init(self):
        """Test checks in init."""
        with self.assertRaises(utils.CIFARDownloader.CIFARDownloaderException):
            utils.CIFARDownloader.CIFARDownloader(cifar_size=3)

    def test_download(self):
        """Test download logic."""
        with temp_dir_context() as tmp_dir:
            downloader = utils.CIFARDownloader.CIFARDownloader(cifar_size=10, local_dir=tmp_dir)

            # No data: try to download, but fail
            with mock.patch.object(downloader, 'download_url', 'http://should_fail_host'), \
                 self.assertRaises(utils.CIFARDownloader.CIFARDownloaderException):
                downloader.get_data()

            downloader._download = mock.MagicMock()
            # No data, but broken archive present: attempt to re-download, but fail
            with open(os.path.join(downloader.local_dir, self.cifar_archive.format(10)), 'w') as archive_handle:
                archive_handle.write('incomplete_data')
                with self.assertRaises(utils.CIFARDownloader.CIFARDownloaderException):
                    downloader.get_data()
                self.assertEquals(downloader._download.call_count, 1)


class TestFilesystem(unittest.TestCase):
    """Test filesystem utility functions."""
    def test_tmp_fir_context(self):
        """Test temp_dir_context path creation and removal."""
        with temp_dir_context() as tmp_dir:
            self.assertTrue(os.path.exists(tmp_dir))
        self.assertFalse(os.path.exists(tmp_dir))

    def test_makedirs(self):
        """Test creation of nested directories."""
        with temp_dir_context() as tmp_dir:
            test_tmp_dir = os.path.join(tmp_dir, 'test_mkdir')
            self.assertFalse(os.path.exists(test_tmp_dir))
            utils.filesystem.make_dir(test_tmp_dir)
            self.assertTrue(test_tmp_dir)

    def test_safe_delete(self):
        """Test deletion of existent/non-existent paths."""
        with temp_dir_context() as tmp_dir:
            utils.filesystem.remove_all(tmp_dir, ignore_errors=False)
            with self.assertRaises(FileNotFoundError):
                utils.filesystem.remove_all(tmp_dir, ignore_errors=False)
            utils.filesystem.remove_all(tmp_dir, ignore_errors=True)
