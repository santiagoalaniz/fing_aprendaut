#!/bin/bash

# URL of the ZIP file to download
zip_url="https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"

# Name of the downloaded ZIP file
zip_file="data.zip"

# Directory where the ZIP file will be downloaded
download_dir="."

# Directory where the contents of the ZIP file will be extracted
extract_dir="./assets"

# Download the ZIP file
wget "$zip_url" --quiet -O "$download_dir/$zip_file"

# Unzip the downloaded ZIP file
unzip -o "$download_dir/$zip_file" -d "$extract_dir" 

# Remove the downloaded ZIP file
rm "$download_dir/$zip_file"

echo "ZIP file downloaded, extracted, and deleted."
