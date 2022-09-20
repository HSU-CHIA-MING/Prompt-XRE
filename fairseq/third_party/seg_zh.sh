lang=${1}

if [ ${lang} == "1" ]; then
	for i in {49..90}
	do
		echo zh/zh_part_${i}.txt
		python zh_segmentation.py --input zh/zh_part_${i}.txt --output zh_seg/zh_part_${i}.txt
	done
elif [ ${lang} == "2" ]; then
	for i in {176..180}
	do
		echo zh/zh_part_${i}.txt
		python zh_segmentation.py --input zh/zh_part_${i}.txt --output zh_seg/zh_part_${i}.txt
	done
elif [ ${lang} == "3" ]; then
	for i in {261..267}
	do
		echo zh/zh_part_${i}.txt
		python zh_segmentation.py --input zh/zh_part_${i}.txt --output zh_seg/zh_part_${i}.txt
	done
fi
