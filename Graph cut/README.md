source.cpp is the c++ example code of Graph Cut Optimization. 

1. Make a new project. You need ../gco and configure it in Visual Studio or anything else. Of course, there is an example.cpp in ../gco. If you want to learn Graph Cut, plz read the example.cpp. I just used the class and function in ../gco.
2. Figure out how the Graph Cut work. There are several blogs may help you on the Internet. You just need to search Graph Cut in baidu or google. It will help you understand the code. The main class I used is GCoptimizationGeneralGraph which you can find in source.cpp. The data term is the negative log of the vote results of different features like SC, IM, GCO... We have mentioned it in our paper. There is a parameter of smooth term, you should change it to get the best result. The value may range from 0.001 to 10. Try it and good luck. 
3. if you want to run the code you should organize files like:

input file:

in test file, 0-6 present 7 features(results of network), gt is groundtruth, prem is the needed files about neighbour triangle meshes.

config.txt: num of label

index.txt: one line present a kind of feature combination,
	   as 0011001 present the combination of three features

output file:

result_127.txt: the average of feature combination in index.txt

result_0011001.txt: the average result of test 1-5

test/allResult.txt: the voteResult of all model in test

test/gcResult.txt: the result after graph cut

Have a good day!
