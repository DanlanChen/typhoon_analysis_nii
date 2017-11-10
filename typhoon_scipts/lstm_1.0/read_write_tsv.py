import csv
from check_land_sea import get_value,gt
def read_tsv(file_name):
	rows=[]
	with open(file_name,'rb') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for row in tsv_reader:
			#print row.type
		    	yy = row[0]
		    	mm = row[1]
		    	dd = row[2]
		    	hh = row[3]
		    	typhoon_type = row[4]
		    	lat = row[5]
		    	lon = row[6]
		    	intensity = row[7]
		    	landorsea = get_value(lon,lat,gt)
                        # print len(row)
                rows.append([yy,mm,dd,hh,typhoon_type,lat,lon,intensity,landorsea])
	return rows
def write_tsv(file_name,rows):
	with open(file_name,'rb') as tsv_file:
		tsv_writer = csv.writer(tsv_file,delimiter = '\t')
		tsv_writer.writerows(rows)

# file_name = '/fs9/danlan/typhoon/data/track/201601.itk'
# read_tsv(file_name)
