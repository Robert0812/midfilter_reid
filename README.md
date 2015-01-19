# midfilter_reid
Matlab code for our CVPR 2014 work on learning mid-level filters for person
re-identification. 

Created by [Rui Zhao](www.ee.cuhk.edu.hk/~rzhao), on May 20, 2013. 

##Summary
In this package, you can find an updated version of MATLAB code for the
following paper:
- Rui Zhao, Wanli Ouyang, and Xiaogang Wang. Unsupervised Salience Learning for
Person Re-identification. In CVPR 2014. 

##Install
- Download [CUHK01
dataset](https://docs.google.com/forms/d/1MF0gAXWKeO1hpsuHlSpPBS8D5JR-r-QOPtdUoFQJONo/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA), and put it into directory ./dataset/campus/
- Compile [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/): and put it to
directory ./code/libsvm
- [GACtoolbox](https://github.com/waynezhanghk/gactoolbox): the gactoolbox in
subfolder ./code/gactoolbox is a revised version. Using the original repository
may cause problem. 
- [DenseFeat](https://github.com/Robert0812/dense_feat): we use the dense color
and SIFT feature, the code for extracting dense features is preliminarily cloned
to ./code/densefeat
- PatchMatch: the patch matching code in ./code/patchmatch/rowcolop_core.cpp is
clone from Dahua Lin's [Statistical Learning
Toolbox](http://www.mathworks.com/matlabcentral/fileexchange/12333-statistical-learning-toolbox), and you need to mex it for using patch match functions. 
- [Structural
SVM](http://www.robots.ox.ac.uk/~vedaldi/code/svm-struct-matlab.html): we only
provide a preliminarily compiled lib for 64-bit windows in ./code/rsvm/, for
other platforms, you are referred to Prof. Andrea Vedaldi's webpage to download the
full code to compile. 

##Demos
One demo is available:
- demo_midfilter_cuhk01.m : perform evaluation on the [CUHK01
dataset](https://docs.google.com/forms/d/1MF0gAXWKeO1hpsuHlSpPBS8D5JR-r-QOPtdUoFQJONo/viewform?formkey=dF9pZ1BFZkNiMG1oZUdtTjZPalR0MGc6MA)

##Remarks
- This implementation is a little different than the original version in the
training / testing partition, so that the result may vary a little. If you use
the default parameter settings, you are suppose to get 33.3% rank-1 matching
rate for only one-trial testing.
- The training / testing partition is generated following the approach
[SDALF](http://www.lorisbazzani.info/code-datasets/sdalf-descriptor/) 
- This demo was tested on MATLAB (R2012b), 64-bit Windows, Intel Xeon 3.33 GHz CPU
- Intermediate cache data would take up to 26GB disk memory 
- Memory cost of demo on the CUHK01 dataset would be around 40GB. 

##Citing our work
Please kindly cite our work in your publications if it helps your research:

	@inproceedings{zhao2014learning,
	    title = {Learning Mid-level Filters for Person Re-identification},
 	    author={Zhao, Rui and Ouyang, Wanli and Wang, Xiaogang},
	    booktitle = {IEEE Conference on Computer Vision and Pattern
		Recognition (CVPR)},
	    year = {2014}
	}

##License

	Copyright (c) 2013, Rui Zhao
	All rights reserved. 

	Redistribution and use in source and binary forms, with or without 
	modification, are permitted provided that the following conditions are 
	met:
    		* Redistributions of source code must retain the above copyright 
      		  notice, this list of conditions and the following disclaimer.
    		* Redistributions in binary form must reproduce the above copyright 
      		  notice, this list of conditions and the following disclaimer in 
      		  the documentation and/or other materials provided with the distribution
   
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 	
	LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
	CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
	SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
	INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
	POSSIBILITY OF SUCH DAMAGE.
