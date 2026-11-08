"""Failure and reuse checks for pinned source downloads."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phonetics.source_preparation.download import download_sounds


def archive_bytes(name="ma1.mp3"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, b"fixture audio container")
    return output.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}

    def geturl(self):
        return "https://www.chinese-lessons.com/sounds/Mandarin_sounds.zip"


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / "sounds"
        self.data = archive_bytes()
        self.pin = hashlib.sha256(self.data).hexdigest()

    def test_success_then_verified_reuse_without_network(self):
        with patch("urllib.request.urlopen", return_value=Response(self.data)) as fetch:
            download_sounds(self.target, expected_sha256=self.pin)
            self.assertEqual(fetch.call_args.args[0].get_header("User-agent"), "curl/8.0.0")
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            self.assertEqual(download_sounds(self.target, expected_sha256=self.pin), self.target)
        self.assertEqual((self.target / "Mandarin_sounds.zip").read_bytes(), self.data)
        self.assertEqual(list(self.root.iterdir()), [self.target])

    def test_corrupt_existing_archive_is_not_replaced(self):
        self.target.mkdir()
        (self.target / "Mandarin_sounds.zip").write_bytes(b"existing")
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(ValueError, "artifact set"):
                download_sounds(self.target, expected_sha256=self.pin)
        self.assertEqual((self.target / "Mandarin_sounds.zip").read_bytes(), b"existing")

    def test_failed_payloads_leave_no_published_or_partial_file(self):
        for data, length, pin in ((b"<html>error</html>", None, self.pin),
                                  (self.data, len(self.data) + 1, self.pin),
                                  (archive_bytes("../unsafe.mp3"), None, None),
                                  (b"not a zip", None, None)):
            with self.subTest(length=length, data=data[:10]):
                expected = pin or hashlib.sha256(data).hexdigest()
                with patch("urllib.request.urlopen", return_value=Response(data, length)):
                    with self.assertRaises((ValueError, zipfile.BadZipFile)):
                        download_sounds(self.target, expected_sha256=expected)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_network_error_cleans_partial(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("fixture timeout")):
            with self.assertRaises(TimeoutError):
                download_sounds(self.target, expected_sha256=self.pin)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
