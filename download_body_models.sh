DIR="support_data"
FILE_ID="1gS-i-P7Qzv1yYQr2jbAw2QGCW208zdit"
FILE_NAME="body_model.zip"

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
