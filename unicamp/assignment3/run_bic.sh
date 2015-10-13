#!/bin/bash
mkdir bic_results
cd bic_results

cont=0

for file in ~/Desktop/mlearning/unicamp/assignment3/ppm/*.ppm; do
	f=$(basename "$file")
	filename="${f%.*}"
 	
	let cont=cont+1

	echo $cont	
	~/bic/source/bin/generatebic $file $filename
done
