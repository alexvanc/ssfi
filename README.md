# SSFI Introduction



SSFI [1] is a Statement-level Software Fault Injection tool, which is able to inject 12 different types of software faults into software systems that can be compiled into Java Bytecode. 

Software faults can be permanent or transient. Thus SSFI provides two different fault activation modes for each type of fault, namely always activation mode and random activation mode. In the always mode, the injected fault is activated every time the code snippet is run. In the random mode, the injected fault is activated when the code snippet is run for the nth time (n is chosen at random if not specified) to simulate transient software faults. SSFI is highly configurable with a simple configuration file specifying which type of fault will be injected to which package/class/method/variables/code blocks with which activation mode. By default, SSFI randomly chooses a combination of the fault type, fault location, and activation mode for the fault.

The following picture gives an overview of SSFI’s structure, which consists of 4 components: Bytecode Parser, Config Parser, Fault Weaver and Converter. SSFI takes a bytecode-based runnable file and a configuration file as inputs, and outputs the modified bytecode- based runnable file with the fault specified in the configuration file. Bytecode Parser leverages Soot [2] framework to parse the bytecode into a Jimple intermediate representation (IR), which is a type of three-address code. The Config Parser interprets the configuration file and determines the fault type,fault location and fault activation mode for the fault to be injected. The Fault Weaver contains transformation rules for each fault type in Table I, and makes the corresponding changes to the Jimple IR based on the fault type, fault location and fault activation mode. It outputs the Jimple IR with the specified fault being injected. Fault Weaver also scans the program’s code automatically to decide which types of faults can be injected (i.e., are valid) into each program statement. Finally, the modified Jimple IR is converted by the Converter, which utilizes Soot to convert Jimple IR to bytecode, into a bytecode-based runnable file with the fault injected.
![SSFI workflow](resource/ssfi_workflow.jpg)

The fault types are listed in the following table.

![SSFI fault types](resource/fault_types.jpg)

# Guide
## build
mvn clean && mvn compile && mvn package

## run
 java -cp ssfi.jar com.alex.ssfi.Application config.yaml

# Reference
1. Yang, Yong, et al. "How far have we come in detecting anomalies in distributed systems? an empirical study with a statement-level fault injection method." 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2020.
2. R.Valle ́e-Rai,P.Co,E.Gagnon,L.Hendren,P.Lam,andV.Sundaresan, “Soot: A java bytecode optimization framework,” in CASCON First Decade High Impact Papers. IBM Corp., 2010, pp. 214–224.

