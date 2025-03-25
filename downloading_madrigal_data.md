# How to download Madrigal GNSS data from the Madrigal website to BlueBear project area
## Link to Madrigal

## Download globaldownload.py
* Get globaldownload.py from this repository

## Commands to use before downloading the Madrigal data
* screen (use this so it doesn't disconnect from terminal)
* module load bluebear
* module load Python/3.9.5-GCCcore-10.3.0
* pip install madrigalWeb

## Get the GNSS data for given dates
* python3 globalDownload.py --url="http://cedar.openmadrigal.org" --user_fullname="Jack Brown" --user_email="jxb1605@student.bham.ac.uk" --outputDir="." --user_affiliation="University of Birmingham" --format="hdf5" --inst=8000 --kindat=3505 --verbose --startDate
=05/09/2024 --endDate=05/13/2024

## Get out of screen
* CTRL + A    d

## Check screen
* screen -r
