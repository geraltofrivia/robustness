#!/usr/bin/env bash
echo "Shall get you up and running in no time"

echo "Making local shell files executable"


echo "Lets first clone mytorch. We'll be using it rather unsparingly."
mkdir mytorch
git clone https://github.com/geraltofrivia/mytorch.git mytorch
cd mytorch
chmod +x setup.sh
./setup.sh
cd ..

echo "TODO: Add data download things here too!"
