import os
import unittest
import unittest.mock as mock

import numpy as np

import utils.CIFARDataset
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
            # No data, but broken archive present: attempt to re-download, and fail
            with open(os.path.join(downloader.local_dir, self.cifar_archive.format(10)), 'w') as archive_handle:
                archive_handle.write('incomplete_data')
                with self.assertRaises(utils.CIFARDownloader.CIFARDownloaderException):
                    downloader.get_data()
                self.assertEqual(downloader._download.call_count, 1)


class TestDataset(unittest.TestCase):
    """Tests related to CIFAR dataset container class."""
    def test_init(self):
        """Test sanity check in init."""
        with self.assertRaises(utils.CIFARDataset.CIFARDatasetException):
            utils.CIFARDataset.CIFARDataset(cifar_size=3)

    def test_load(self):
        """Test error handling in the load method."""
        dataset = utils.CIFARDataset.CIFARDataset(cifar_size=10)
        with temp_dir_context() as tmp_dir, self.assertRaises(utils.CIFARDataset.CIFARDatasetException):
            dataset.load(tmp_dir)

    def test_pre_process(self):
        """Test pre-processing for CNN."""
        base_line = 100
        dataset = utils.CIFARDataset.CIFARDataset(cifar_size=10)

        dataset.train_img_categories = np.array([0, 1])
        dataset.train_img_data = base_line + np.random.random((2, 3, 32, 32))
        self.assertTrue(np.all(dataset.train_img_data >= base_line))
        with self.assertRaises(utils.CIFARDataset.CIFARDatasetException):
            dataset.pre_process_for_cnn()
        dataset.test_img_data = dataset.train_img_data
        dataset.test_img_categories = dataset.train_img_categories
        dataset.pre_process_for_cnn()

        self.assertTrue(np.all(dataset.train_img_data <= 1))
        self.assertTrue(np.all(dataset.train_img_data >= 0))
        self.assertEqual(dataset.train_img_data.shape, dataset.test_img_data.shape)


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
