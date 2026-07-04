import json
import unittest

from xquik_export import parse_xquik_export


class XquikExportTest(unittest.TestCase):
    def test_parse_wrapped_json(self):
        raw = json.dumps({"data": [{"full_text": "first row"}]}).encode("utf-8")
        self.assertEqual(parse_xquik_export(raw, "tweets.json"), ["first row"])

    def test_parse_csv(self):
        raw = b"username,text\nalice,second row\n"
        self.assertEqual(parse_xquik_export(raw, "tweets.csv"), ["second row"])

    def test_reject_missing_text(self):
        raw = json.dumps({"data": [{"id": "1"}]}).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "tweet text"):
            parse_xquik_export(raw, "tweets.json")


if __name__ == "__main__":
    unittest.main()
