import csv
import json

def make_json(csvFilePath, jsonFilePath):
    data = {}
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            rows.pop('', None)
            key = rows['PTID']
            rows.pop('PTID', None)

            comb = dict()
            delete = []
            for item in rows.keys():
                index = item.find('_')
                if index == -1 or item[index+1:].isalpha():
                    continue
                if rows[item] == 'True':
                    comb[item[:index]] = str(float(item[index+1:]))
                delete.append(item)
            for item in delete:
                rows.pop(item, None)
            rows.update(comb)
            data[key] = rows
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
csvFilePath = r'clinical.csv'
jsonFilePath = r'clinical.json'
make_json(csvFilePath, jsonFilePath)
