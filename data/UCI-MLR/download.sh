wget -r -np -R "index.html*" https://archive.ics.uci.edu/ml/machine-learning-databases/00442/
rsync -a archive.ics.uci.edu/ml/machine-learning-databases/00442/* ./
rm -r archive.ics.uci.edu
find ./ -name '*.rar' -execdir unar {} \;
