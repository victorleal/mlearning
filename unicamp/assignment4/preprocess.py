import csv

# NUMBER OF ROWS TO LEAD
cont = 0
THRESHOLD = 100
with open("dataset.txt","r") as infile:
	
	content = csv.reader(infile)

	l = []
	i = 0
	last_user = -1
	file_mode = "w"

	# READING DATA FOR EACH USER
	for row in content:

		# IF SAME USER, APPEND ROWS TO LIST
		if(row[0] == last_user or last_user == -1):
			l.append(row)
			last_user = row[0]

		# WHEN USER CHANGES, APPLY THE LEAD AND SAVE TO FILE
		else:
			print("Writing user: "+last_user)
			
			# CALCULATE THE WINDOWS
			for i in range(0, len(l), 5):
				#print l[i][ len(l[i])-1 ]
				l[i].pop( len(l[i])-1 )
				#l[i].pop(6) # removing trailing element

				for j in range(i+1,min(i+THRESHOLD,len(l))):
					
					# user_id and activity must match
					if l[i][0] == l[j][0] and l[i][1] == l[j][1]:
						l[i].append(l[j][3]) # X
						l[i].append(l[j][4]) # Y
						l[i].append(l[j][5]) # Z

			# REMOVING INCOMPLETE LINES
			i = 0
			while i < len(l):
				if len(l[i]) < 6 + 3 * (THRESHOLD-1):
					l.pop(i)
				else:
					i += 1

			# SAVE TO FILE
			with open("dataset_range_5.txt",file_mode) as outfile:

				# At first time, open with "w". Then, open with "a"
				if file_mode == "w": file_mode = "a"

				writer = csv.writer(outfile)
				writer.writerows(l)
				l = []
				outfile.close()
			last_user = -1
			cont += 1
			

	infile.close()
	outfile.close()
	print cont
