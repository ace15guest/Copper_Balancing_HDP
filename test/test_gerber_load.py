import unittest
from unittest.mock import patch, MagicMock
import subprocess
from pathlib import Path
from loading.gerber_load import check_gerber

class TestCheckGerber(unittest.TestCase):
    @patch('loading.gerber_load.subprocess.Popen')
    @patch('loading.gerber_load.Path')
    @patch('loading.gerber_load.time.time', side_effect=[100, 105])
    def test_check_gerber_creates_log_file(self, mock_time, mock_path, mock_popen):
        # Setup
        mock_path.return_value.mkdir.return_value = None
        mock_popen.return_value = MagicMock()

        # Execute
        log_file_name = check_gerber(r"C:\path\to\file.gbr")

        # Verify
        mock_path.assert_called_with(r"Assets\temp")
        mock_path.return_value.mkdir.assert_called_with(exist_ok=True, parents=True)
        self.assertEqual(log_file_name, r"Assets\temp\file.gbr.log")
        mock_popen.assert_called_with('Assets\gerbv\gerbv -x rs274x -o NUL "C:\\path\\to\\file.gbr" 2>"Assets\\temp\\file.gbr.log"', shell=True)

if __name__ == '__main__':
    unittest.main()