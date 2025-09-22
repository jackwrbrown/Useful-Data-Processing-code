# How to download Madrigal GNSS data from the Madrigal website to BlueBear project area
## Link to Madrigal
http://millstonehill.haystack.mit.edu/list

## Download globaldownload.py
* Get globaldownload.py from this repository

## Commands to use before downloading the Madrigal data
* screen (use this so it doesn't disconnect from terminal)
* module load bluebear
* module load Python/3.9.5-GCCcore-10.3.0
* pip install madrigalWeb

## Get the GNSS data for given dates
* python3 globalDownload.py --url="http://cedar.openmadrigal.org" --user_fullname="Jack Brown" --user_email="jxb1605@student.bham.ac.uk" --outputDir="." --user_affiliation="University of Birmingham" --format="hdf5" --inst=8000 --kindat=3505 --verbose --startDate=05/09/2024 --endDate=05/13/2024

## Get out of screen
* CTRL + A    d

## Check screen
* screen -r


## Useful information about Madrigal GNSS data:
* Experiment Parameters:
* instrument: World-wide GPS Receiver Network
* instrument code(s): 8000
* kind of data file: Line of sight TEC data
* kindat code(s): 3505

* Cedar file name: los_20170603.001.h5 (example file name for June 3rd 2017)
* instrument category: Distributed Ground Based Satellite Receivers (this is where the data you're getting is from)

### Data Parameters:
* YEAR: Year (universal time), units: y
* MONTH: Month (universal time), units: m
* DAY: Day (universal time), units: d
* HOUR: Hour (universal time), units: h
* MIN: Minute (universal time), units: m
* SEC: Second (universal time), units: s
* RECNO: Logical Record Number, units: N/A
* KINDAT: Kind of data, units: N/A
* KINST: Instrument Code, units: N/A
* UT1_UNIX: Unix seconds (1/1/1970) at start, units: s
* UT2_UNIX: Unix seconds (1/1/1970) at end, units: s
* PIERCE_ALT: Pierce Point Altitude, units: km
* GPS_SITE: Four letter GPS receiver code, units: n/a
* SAT_ID: Satellite id, units: N/A
* GDLATR: Reference geod latitude (N hemi=pos), units: deg
* GDLONR: Reference geographic longitude, units: deg
* LOS_TEC: Line-of-sight integrated electron density, units: los_tec
* DLOS_TEC: Error in Line-of-sight integrated electron density, units: los_tec
* TEC: Vertically integrated electron density, units: tec
* AZM: Mean azimuth angle (0=geog N;90=east), units: deg
* ELM: Elevation angle (0=horizontal;90=vert), units: deg
* GDLAT: Geodetic latitude of measurement, units: deg
* GLON: Geographic longitude of measurement, units: deg
* REC_BIAS: GPS receiver bias, units: TECu
* DREC_BIAS: Error in GPS receiver bias, units: TECu
