lang=${1}

if [ ${lang} == "zh" ]; then
	cd zh
	for i in {1..267}
	do
		wget --http-user=xhaoran2 --http-passwd=Qweruiopad34  https://oscar-prive.huma-num.fr/2109/packaged/zh/zh_part_${i}.txt.gz
	done
elif [ ${lang} == "en" ]; then
	cd en
	for i in {415..543}
	do
		wget --http-user=xhaoran2 --http-passwd=Qweruiopad34  https://oscar-prive.huma-num.fr/2109/packaged/en/en_part_${i}.txt.gz
	done
elif [ ${lang} == "ru" ]; then
	cd ru
	for i in {536..543}
	do
		wget --http-user=xhaoran2 --http-passwd=Qweruiopad34  https://oscar-prive.huma-num.fr/2109/packaged/ru/ru_part_${i}.txt.gz
	done
fi
