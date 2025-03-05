import unittest
import json
import os
from arxiv_recommender.utils.json_handler import load_json, save_json

TEST_JSON_FILE = "test_favorite_papers.json"

class TestJSONHandler(unittest.TestCase):
    
    def setUp(self):
        """Setup sample data before each test."""
        self.sample_data = [{"title": "Sample Paper", 
                             "abstract": "Test abstract."}]
    
    def tearDown(self):
        """Cleanup test JSON file after tests."""
        if os.path.exists(TEST_JSON_FILE):
            os.remove(TEST_JSON_FILE)

    def test_save_json(self):
        """Test saving data to JSON file."""
        save_json(TEST_JSON_FILE, self.sample_data)
        self.assertTrue(os.path.exists(TEST_JSON_FILE))

        with open(TEST_JSON_FILE, "r", encoding="utf-8") as file:
            saved_data = json.load(file)
        
        self.assertEqual(saved_data, self.sample_data)

    def test_load_json(self):
        """Test loading data from JSON file."""
        save_json(TEST_JSON_FILE, self.sample_data)
        loaded_data = load_json(TEST_JSON_FILE)
        self.assertEqual(loaded_data, self.sample_data)

    def test_load_nonexistent_json(self):
        """Test handling missing file."""
        with self.assertRaises(IOError):
            load_json("non_existent.json")

    def test_load_invalid_json(self):
        """Test handling corrupt JSON files."""
        with open(TEST_JSON_FILE, "w", encoding="utf-8") as file:
            file.write("Invalid JSON data")

        with self.assertRaises(json.JSONDecodeError):
            load_json(TEST_JSON_FILE)

if __name__ == "__main__":
    unittest.main()
