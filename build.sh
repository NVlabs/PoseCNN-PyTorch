cd lib/layers/;
python3 setup.py build develop;
cd ../utils;
python3 setup.py build_ext --inplace

