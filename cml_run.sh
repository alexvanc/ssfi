#! /bin/bash
mvn clean
mvn package
rm ../llfi*.jar
rm ../WorkBench.class
cp target/llfi*dependen*.jar ../
cd ..
java -cp llfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.llfi.App -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true -f jimple WorkBench
java -cp llfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.llfi.App -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true WorkBench
#java -cp sootclasses-trunk-jar-with-dependencies.jar soot.Main -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true -f jimple -d output2 WorkBench
cp ssfi/target/classes/WorkBench.class .
cd -
