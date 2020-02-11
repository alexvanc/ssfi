#! /bin/bash
word_dict=('hello' 'my' 'name' 'is' 'ssfi' 'please' 'specify' 'the' 'word_dict' 'list' 'by' 'yourself')
lines=$1
while [ $lines -ge 1 ] 
do
    for word in ${word_dict[@]}
    do
        echo -n "$word ">> input_data.txt
    done
    echo "" >> input_data.txt
    lines=$[lines-1]
done
    
