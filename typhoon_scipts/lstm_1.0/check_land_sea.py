from osgeo import gdal
from osgeo import osr
from PIL import Image
import numpy,os
from osgeo.gdalconst import *
import struct
import sys
# from read_write_tsv import read_tsv, write_tsv
file_name = '/Users/DanlanChen/Desktop/lmtgsh3a.tif'
ds = gdal.Open(file_name)
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5] 
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3] 
band = ds.GetRasterBand(1)
bandtype = gdal.GetDataTypeName(band.DataType) 
import csv

def read_tsv(file_name):
	
	with open(file_name,'rb') as tsv_file:
		tsv_reader = csv.reader(tsv_file, delimiter='\t')
		rows=[]
		count = 0
		for row in tsv_reader:
			# print row
			#print row.type
			yy = row[0]
			mm = row[1]
			dd = row[2]
			hh = row[3]
			typhoon_type = row[4]
			lat = float(row[5])
			lon = float(row[6])
			# if lon>180.0:
			# 	print file_name,'longtitude large than 180'
			intensity = row[7]
			landorsea =row[-4]
			# try:
			# 	landorsea = get_value(lon,lat,gt)
			# except:
			# 	print file_name,'landfall',count,'landorseawrong'
			# 	# break

			if landorsea == 1:
				print file_name,'landfall',count
			rows.append([yy,mm,dd,hh,typhoon_type,lat,lon,intensity,landorsea])
			count +=1
	return rows
def write_tsv(file_name,rows):
	with open(file_name,'wb') as tsv_file:
		tsv_writer = csv.writer(tsv_file,delimiter = '\t')
		tsv_writer.writerows(rows)
def pt2fmt(pt):
	fmttypes = {
		GDT_Byte: 'B',
		GDT_Int16: 'h',
		GDT_UInt16: 'H',
		GDT_Int32: 'i',
		GDT_UInt32: 'I',
		GDT_Float32: 'f',
		GDT_Float64: 'f'
		}
	return fmttypes.get(pt, 'x')
def get_geo(Xpixel,Yline,GT):

	Xgeo = GT[0] + Xpixel*GT[1] + Yline*GT[2]
	Ygeo = GT[3] + Xpixel*GT[4] + Yline*GT[5]
	return Xgeo,Ygeo


def get_px_py(Xgeo,Ygeo,GT):
	px = int((GT[5]*Xgeo - GT[2]*Ygeo - GT[0]*GT[5] + GT[3] * GT[2])/(GT[1] *GT[5] - GT[2]* GT[4]))
	py = int((GT[4] * Xgeo - GT[1]*Ygeo - GT[0]*GT[4] + GT[1] *GT[3])/(GT[2]*GT[4] - GT[1] *GT[5]) )
	return px,py
def read_pixel_v(px,py):
	structval = band.ReadRaster(int(px), int(py), 1,1, buf_type = band.DataType )

	fmt = pt2fmt(band.DataType)

	intval = struct.unpack(fmt , structval)

	return round(intval[0],2) #int
def read_pixel_v_2(file):
	im = Image.open(file)
	imarray = numpy.array(im)
	imarray = numpy.transpose(imarray)
	# print imarray.shape
	return imarray



# Xgeo,Ygeo = get_geo(43200,21600,gt)
# print Xgeo,Ygeo
# lot = 138.500000
# lat = 20.0
def get_value(lot,lat,gt):
	px,py = get_px_py(lot,lat,gt)
	# print px,py
	if px >= 43200 or py >= 21600:
		print px,'error',lot,lat
		print py,'error',lot,lat
		px = min(px,43199)
		py = min(py,21599)
	v = read_pixel_v(px,py)
	# print v
	if v == 129.0:
		b = 0
	else:
		b = 1
	# # else:
	# # 	print v
	return b
def main():
	old_path = '/Volumes/Danlan/nii_typhoon_data/fs9_2/danlan/typhoon/data/track'
	new_path = '/Volumes/Danlan/nii_typhoon_data/new_track/'
	for subdir,dirs, files in os.walk(old_path):
		for file in files:
			file_path = os.path.join(subdir,file)
			new_file_path = os.path.join(new_path,file)
			rows = read_tsv(file_path)
			# print len(rows)
			# write_tsv(new_file_path,rows)
if __name__ == '__main__':
	main()
	# lat,lot = 36.108874,140.352797
	# b = get_value(lot,lat,gt)
	# print b
	# imarray = read_pixel_v_2(file_name)
	# print imarray[38442][6466]



