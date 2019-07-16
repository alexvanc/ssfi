#cd $INPUT
#mv $TARGET $OUTPUT/$TARGET
#mv $OUTPUT/$TARGET.bak $TARGET
#java -cp test.jar com.alex.halo.App >/tmp/runNormal.txt 2>/tmp/runError.txt & echo $!
java -cp test.jar com.alex.halo.App >/tmp/runNormal.txt 2>/tmp/runError.txt
# rm test.jar

