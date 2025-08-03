library(auk)
library(dplyr)

# Choose Observations Metadata file (obtained from downloading a data archive through the ebird database)
metadata_path <- file.choose()

# Choose Checklist Metadata file (usually, the sampling.txt file obtained from downloading a data archive through the ebird database)
sampling_path <- file.choose()

# Output file name
output_file = "/Volumes/My Passport/usa_data_ebird/extracted_data/test_summer_2024/ebird_data.csv"
sampling_file = "/Volumes/My Passport/usa_data_ebird/extracted_data/test_summer_2024/sampling_ebird.csv"

ebd <- auk_ebd(metadata_path, file_sampling=sampling_path) # Convert observations and checklist metadata files to AUK object format

ebd_filters <- auk_complete(ebd) # Remove non-complete checklists
ebd_filtered <- auk_filter(ebd_filters, file = output_file, file_sampling = sampling_file, overwrite=TRUE)