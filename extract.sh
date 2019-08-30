#! /bin/bash
cd $1
for component in ./*; do
    if [ -d $component ]
    then
        cd $component
        for package in ./*; do
            filename=$(basename -- "$package")
            extension="${filename##*.}"
            name="${filename%.*}"
            mkdir $name
            mv $filename $name/
            cd $name
            jar -xf $filename
            rm $filename
            cd ..
            done
        cd ..
    else
        echo "Non-components found!"
    fi
done
