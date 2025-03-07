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
                val = item[index+1:]
                par = item[:index]
                if index == -1 or val.isalpha():
                    continue
                
                val = val.replace('_', '.')
                
                if rows[item] == 'True':
                    comb[par] = str(float(val))
                delete.append(item)
            for item in delete:
                rows.pop(item, None)
                
            rows.update(comb)
            myKeys = list(rows.keys())
            myKeys.sort()    
            rows = {i: rows[i] for i in myKeys}
            data[key] = rows
    
    count = {}
    for sub in data.keys():
        subject = data[sub]
        for param in subject:
            count[param] = {}
            value = subject[param]
            if value not in count[param].keys():
                count[param][value] = 1
            else:
                count[param][value] += 1

    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
csvFilePath = r'clinical.csv'
jsonFilePath = r'clinical.json'
make_json(csvFilePath, jsonFilePath)
