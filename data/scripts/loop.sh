#!/bin/bash

echo -n "Company: "
read company

PS3='Document type: '
options=("AR" "20F" "SR")
select opt in "${options[@]}"
do
    type=$opt
    case $opt in
        "AR")
            echo "Selected AR..."
            break
            ;;
        "20F")
            echo "Selected 20F..."
            break
            ;;
        "SR")
            echo "Selected SR"
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

echo -n "URL: "
read url
echo -n "start year: "
read start
echo -n "end year: "
read end 

for year in $(seq $start $end);
do
   year_url=${url//yyyy/$year}
   next_year=$((year+1))
   year_url=${year_url/YYYY/$next_year}
   path=./${company}
   file_name=${type}_${year}.pdf
   full_path=${path}/data/${file_name}
   echo "Using URL: $year_url to save in $file_name"
   curl -f $year_url -o $full_path -A "User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.2.12) Gecko/20101026 Firefox/3.6.12"

done
