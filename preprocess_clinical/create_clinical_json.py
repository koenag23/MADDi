import csv
import json

def csv_to_json(csv_filepath, json_filepath):
    data = {}

    # Read the CSV file
    with open(csv_filepath, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)  # Read the header row
        
        for row in csv_reader:
            key = row[1]  # Use the second column as the key
            data[key] = {headers[i]: row[i] for i in range(len(row)) if i > 1}  # Store other values

    # Write to a JSON file
    with open(json_filepath, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)


# Example usage
csv_to_json('clinical.csv', 'clinical.json')
