from os import listdir
from os.path import isfile, join

mypath = "/home/victor/Desktop/mlearning/unicamp/assignment3/bic_results/"

files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
features = []
cont = 0

summarized = open("summarized_bic.txt", "w")

for f in files:
	cont+=1
	print cont

	txt = open(join(mypath, f))
	lines = [ l for l in txt.readlines() ]

	l = lines[1]
	l = l[1:]
	l = l[:len(l)-1]

	cont2 = 0
	for c in l:
	    summarized.write(str(c))
	    if cont2 < len(l)-1:
	        summarized.write(",")
	    cont2+=1

	summarized.write("\n")
	txt.close()
	
summarized.close()
