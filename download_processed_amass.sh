DIR="data_amass_avatarposer_split"
FILE_ID="1M9n9QUVN2yr2DiuVY_wMxyrK0iy-ZNsg"
FILE_NAME="data_amass_avatarposer_split.zip"

if [ ! -d "$DIR" ]; then
  echo "'$DIR' directory does not exist. Downloading the file..."
  gdown $FILE_ID -O "$FILE_NAME" 
  unzip "$FILE_NAME" -d "$DIR"
  cd "$DIR"
  cd "$DIR"
  mv * ..
  cd ..
  rm -r "$DIR"
  cd ..
  rm "$FILE_NAME"
else 
  echo "'$DIR' directory already exists. No need to download."
fi
