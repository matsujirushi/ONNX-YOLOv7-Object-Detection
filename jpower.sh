#!/bin/bash

power_sensor_path=/sys/bus/i2c/devices/6-0040/iio:device0
# igpu_path=/sys/class/devfreq/57000000.gpu

echo -n "TIME "
for i in {0..2} ; do
  cat ${power_sensor_path}/rail_name_${i} | tr -d '\n'
  echo -n " "
done
# echo -n "FREQ_GPU "
echo

while true
do
  echo `date +'%s.%3N'` | tr -d '\n'
  echo -n " "
  for i in {0..2} ; do
    cat ${power_sensor_path}/in_power${i}_input | tr -d '\n'
    echo -n " "
  done
  # cat ${igpu_path}/cur_freq | tr -d '\n'
  echo

  sleep 0.01
done
