#!/usr/bin/env bash
echo "Shall get you up and running in no time"

echo "It's important that you have activated your coda/pyenv environment beforehand. If not, please exit and do that first."
while true; do
    read -p "Do you wish to exit this program? : " yn
    case $yn in
        [Nn]* ) make install; break;;
        [Yy]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

#echo "Lets first clone mytorch. We'll be using it rather unsparingly."
#mkdir mytorch
#git clone https://github.com/geraltofrivia/mytorch.git mytorch
#cd mytorch
#chmod +x setup.sh
#./setup.sh
#cd ..


# We'll need large spacy model which probably doesn't ship out of the box.
python -m spacy download en_core_web_lg
echo "TODO: Add data download things here too!"
