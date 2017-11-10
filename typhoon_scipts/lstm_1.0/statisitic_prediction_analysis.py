import csv,config
csv_file = config.csv_path
dicts={2:0,3:0,4:0,5:0,6:0,7:0}
wrong_dicts={2:0,3:0,4:0,5:0,6:0,7:0}
ratio_dicts={}
with open(csv_file,'rb') as csvfile:
	reader = csv.reader(csvfile,delimiter = ',')
	next(reader, None)
	for row in reader:
		dicts[int(row[-1])] +=1
		if row[-2] != row[-1]:
			wrong_dicts[int(row[-1])] +=1
print dicts
print wrong_dicts
for k,v in dicts.iteritems():
	ratio_dicts[k] = 1.0*wrong_dicts[k]/v
print ratio_dicts
