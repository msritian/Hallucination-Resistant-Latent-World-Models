"""Append-only CSV log that adds columns as new keys appear."""
import csv
from pathlib import Path


class CSVLog:
	def __init__(self, path: Path):
		self.path = path
		self.fields = None
		if path.exists():
			with open(path) as f:
				header = f.readline().strip()
				self.fields = header.split(",") if header else None

	def write(self, row: dict):
		new = [k for k in row if self.fields is None or k not in self.fields]
		if new:  # (re)write the header with any new columns, keeping earlier rows
			rows = list(csv.DictReader(open(self.path))) if self.fields is not None else []
			self.fields = (self.fields or []) + new
			with open(self.path, "w", newline="") as f:
				w = csv.DictWriter(f, self.fields)
				w.writeheader()
				w.writerows(rows)
		with open(self.path, "a", newline="") as f:
			csv.DictWriter(f, self.fields).writerow(row)
