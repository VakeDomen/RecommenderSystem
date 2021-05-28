if pip3 show pandas ; [ $? -ne 0 ];then
  echo "Pandas not installed! Installing it now!"
  pip3 install pandas
fi

if pip3 show sklearn ; [ $? -ne 0 ];then
  echo "sklearn not installed! Installing it now!"
  pip3 install sklearn
fi

if pip3 show scikit-learn ; [ $? -ne 0 ];then
  echo "scikit-learn not installed! Installing it now!"
  pip3 install scikit-learn
fi

if pip3 show scipy ; [ $? -ne 0 ];then
  echo "scipy not installed! Installing it now!"
  pip3 install scipy
fi

if pip3 show tqdm ; [ $? -ne 0 ];then
  echo "TQDM not installed! Installing it now!"
  pip3 install tqdm
fi
echo "Done!"
