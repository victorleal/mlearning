import csv

# NUMBER OF ROWS TO LEAD
THRESHOLD = 100
STD_JUMP = 10
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
			i = 0
			while(i < len(l)):
				l[i].pop(6) # removing trailing element

				for j in range(i+1,min(i+THRESHOLD,len(l))):
					
					# user_id and activity must match
					if l[i][0] == l[j][0] and l[i][1] == l[j][1]:
						l[i].append(l[j][3]) # X
						l[i].append(l[j][4]) # Y
						l[i].append(l[j][5]) # Z
				
				# Weighted jumping for each class
				if l[i][1] == "Walking":
					i += STD_JUMP*4
				elif l[i][1] == "Jogging":
					i += STD_JUMP
				elif l[i][1] == "Stairs":
					i += int(STD_JUMP/5)
				elif l[i][1] == "Sitting":
					i += STD_JUMP*2	
				elif l[i][1] == "Standing":
					i += STD_JUMP
				elif l[i][1] == "LyingDown":
					i += STD_JUMP	

			# REMOVING INCOMPLETE LINES
			i = 0
			while i < len(l):
				if len(l[i]) < 6 + 3 * (THRESHOLD-1):
					l.pop(i)
				else:
					i += 1

			# SAVE TO FILE
			with open("dataset_ok.txt",file_mode) as outfile:

				# At first time, open with "w". Then, open with "a"
				if file_mode == "w": file_mode = "a"

				writer = csv.writer(outfile,lineterminator="\n")
				writer.writerows(l)
				l = []
				outfile.close()
			last_user = -1
			

	infile.close()
	outfile.close()