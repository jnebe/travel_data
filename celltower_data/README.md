# Quick Notes on the Data

Data consists of samples from November 2022

The data is already processed, that is:

- converted in EPSG:4326 format
- the distance field (in km) has been added
- trips with <100km have been filtered out
- please note the 'frequency' field for each entry, which represents the # of trips

More information about the frequency and way trips were derived:

The 'frequency' field refers to the number of aggregated devices that were recorded doing the trip from start-coord to end-coord at any point in time of Nov 22. A device is registered to a coord if its stay near the corresponding antenna lasts more than 15 minutes. That means that once the device stops at a coord for >15min, this coord will be considered as end-coord and the trip is regarded as ended. The previously registered coord is considered as start-coord.