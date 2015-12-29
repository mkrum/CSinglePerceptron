import csv

f = open("breastcancerdata.csv")
csv_f = csv.reader(f)
write = open("cancer.txt", 'wb')
csv_w = csv.writer(write)
for row in csv_f:
    if "?" not in row:
        row.pop(0)
        if(row[-1] == '2'):
            row[-1] = "benign"
        else:
            row[-1] = "malignant"
        csv_w.writerow(row)
        

