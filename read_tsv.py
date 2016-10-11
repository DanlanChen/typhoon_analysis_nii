import csv
def read_tsv(file_name):
	with open(file_name,'rb') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		for row in tsv_reader:
			# print row.type #type :list
			print row
			break
file_name = '/fs9/danlan/typhoon/data/track/201601.itk'
read_tsv(file_name)
#['2016', '07', '02', '12', '2', '8.300000', '145.100000', '1006.000000', '0.000000', '0.000000', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
