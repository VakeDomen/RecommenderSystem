if pip3 show pandas ; [ $? -ne 0 ];then
  echo "Pandas not installed! Installing it now!"
  pip3 install pandas
fi


if pip3 show tqdm ; [ $? -ne 0 ];then
  echo "TQDM not installed! Installing it now!"
  pip3 install tqdm
fi
echo "Done!"
