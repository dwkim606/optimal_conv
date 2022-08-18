# Source code for **Optimized Privacy-Preserving CNN Inference with Fully Homomorphic Encryption**

## Requirements
1. Go 1.16.6 or higher (<https://go.dev/>)  
- Install command with apt-get:  
```console
apt-get install golang-go
```  
2. Go packages (Go Cryptography \& Lattigo (fork))  
- After installing Go, install packages with following commands: 
```console
go get -u golang.org/x/crypto/...
go get -u github.com/dwkim606/test_lattigo
```  
**CAUTION**: For Lattigo, we must install the <em>forked version</em> with above command (instead of the latest [one](https://github.com/tuneinsight/lattigo))  

3. Python3 with numpy package (required only for checking the precision of CNN classifier)   

## Running the Test  

0. Dataset Preparation: **Necessary** to run the tests  

- Download the data file from the link: https://drive.google.com/drive/folders/1zLTzJ58E_CDtqvnPv8t9YtgkDaHouWWn?usp=sharing  
- Move all folders (Resnet_enc_results, Resnet_plain_data, Resnet_weights, test_conv_data) to the same directory as the source code.  


The following tests are available:   

1. Convolutions: run Baseline and Ours for various number of batches  
- Arguments (in the order of input):
	- conv
	- (3,5,7): width of kernel
	- (0,1,2,3): set number of batches to 4, 16, 64, 256, respectively.
	- (1 to 10): number of test runs
- Example command: convolution with kernel width 3, number of batches 16, and 5 test runs 
```console
go run *.go conv 3 1 5
```  

2. Convolutions followed by ReLU evaluation (and Bootstrapping): run Baseline and Ours  
- Arguments: 
	- convReLU
	- other parts are the same as Convolutions 
- Example command: convolution with kernel width 5, number of batches 4, and 3 test runs, then ReLU evaluation with Bootstrapping 
```console
go run *.go convReLU 5 0 3
```  
3. 20-layer CNN evaluatoin with our method on CIFAR10/CIFAR100 dataset   
- Arguments:
	- resnet
	- (3,5,7): width of kernel
	- (8,14,20): number of layers of CNN
	- (1,2,3): wideness factor
	- (1 to 100 or 1 to 1000): number of tests
	- (true, false): true -> CIFAR100, false -> CIFAR10  
- **List of available arguments**: (given in the paper; other arguments require appropriate weights for CNN)
	- resnet 3 20 1 (1 to 1000) (true/false)
	- resnet 5 20 1 (1 to 100) (true/false)
	- resnet 7 20 1 (1 to 100) (true/false)
	- resnet 5 8 3 (1 to 1000) (true/false)
	- resnet 3 14 3 (1 to 1000) (true/false)
	- resnet 3 20 3 (1 to 1000) (true/false)
- Example command: run CNN with kernel width 3, number of layers 20, widness factor 1, on 10 test inputs on CIFAR10 dataset 
```console
go run *.go resnet 3 20 1 10 false
```  
**CAUTION**: The CNN evaluation test requires at most roughly 100GB of memory.  
- To check the precision of encrypted inference, run compare_final.py with python3, argument (width of kernel, depth, widness, CIFAR100 or not)  
- Example command: check the precision of kernel 3, depth 20, widness 1 CNN inference on CIFAR10 dataset  
```console
python3 compare_final.py 3 20 1 false
``` 
**CAUTION**: Precision check requires the encrypted inference to be performed beforehand.   

## MISC.
One can also generate an executble for a remote server with Linux and AMD cpu via the following command.  
(Modify the command appropriately for other OS and cpu)   
```console
env GOOS=linux GOARCH=amd64 go build -o test_run
```  
It will generate an executable titled "test_run", which can run on the server with appropriate arguments.  
(e.g., for convolution)    
```console
./test_run conv 3 1 5
```  
With this command, one can run the test on any server without Go (given that executable is generated from other device with Go for compile).  
The executable must be in the same directory as data folders (Resnet_enc_results, Resnet_plain_data, Resnet_weights, test_conv_data).
