import csv
import os.path
from typing import List, Any, Dict

from utils.enums import DataType

CSV_FILE = "results.csv"


class csv_handler:
    def __init__(self, folder: str, parameters: List[DataType]):
        self.folder = folder
        self.out_file = open(os.path.join(self.folder, CSV_FILE), 'w', encoding="utf-8", newline='')
        self.writer = csv.writer(self.out_file, delimiter=",", quoting=csv.QUOTE_NONE)
        self.parameters = parameters
        self.writer.writerow([param.value for param in self.parameters])

    def write_row(self, values_dict: Dict[DataType, Any]):
        values = [str(values_dict[param]) for param in self.parameters]
        self.writer.writerow(values)

    def get_current_data(self):
        self.out_file.close()
        self.out_file = open(os.path.join(self.folder, CSV_FILE), 'r', encoding="utf-8")
        reader = csv.reader(self.out_file, delimiter=",", quoting=csv.QUOTE_NONE)
        lines = list(reader)
        self.out_file.close()
        self.out_file = open(os.path.join(self.folder, CSV_FILE), 'a', encoding="utf-8", newline='')
        self.writer = csv.writer(self.out_file, delimiter=",", quoting=csv.QUOTE_NONE)
        return lines

    @staticmethod
    def load_all_data(file_path: str):
        with open(file_path, 'r') as read_file:
            reader = csv.reader(read_file, delimiter=",", quoting=csv.QUOTE_NONE)
            lines = list(reader)
        return lines