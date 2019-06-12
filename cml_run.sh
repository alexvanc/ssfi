#! /bin/bash
mvn clean
mvn package -DskipTests
rm ../ssfi*.jar
rm ../WorkBench.class
cp target/ssfi*dependen*.jar ../
cp target/classes/WorkBench.class ../
cd ..
#java -cp ssfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.ssfi.App -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true -f jimple WorkBench
java -cp ssfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.ssfi.Application /home/alex/work/soot/ssfi/src/main/resources/config.yaml
#java -cp ssfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.ssfi.App -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true WorkBench
#java -cp ssfi-1.0-SNAPSHOT-jar-with-dependencies.jar com.alex.ssfi.App -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true -f jimple WorkBench
#java -cp sootclasses-trunk-jar-with-dependencies.jar soot.Main -cp .:/home/alex/work/soot/ssfi/target/classes -pp -p jb use-original-names:true -f jimple -d output2 WorkBench
cp ssfi/target/classes/WorkBench.class .
cd -
