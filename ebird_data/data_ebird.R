library(auk)
library(dplyr)

metadata_path <- file.choose()
sampling_path <- file.choose()

output_file = "/Volumes/My Passport/usa_data_ebird/extracted_data/test_summer_2024/ebird_data.csv"
sampling_file = "/Volumes/My Passport/usa_data_ebird/extracted_data/test_summer_2024/sampling_ebird.csv"

ebd <- auk_ebd(metadata_path, file_sampling=sampling_path)

ebd <- auk_ebd(sampling_path, file_sampling=metadata_path)

ebd_filters <- auk_complete(ebd)
ebd_filtered <- auk_filter(ebd_filters, file = output_file, file_sampling = sampling_file, overwrite=TRUE)