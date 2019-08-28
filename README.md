# Face-alignment-Trees
This is the C++ implement of the paper: Face Detection, Pose Estimation, and Landmark Localization in the Wild

I write this c++ code to speed up the original version. However, it still needs lots of time to process a single image. You can adjust some parameter setting but it don't improve the runtime performance a lot in fact. Maybe you should take advantage of other methods instead. Here is the test result:

![](https://github.com/goodluckcwl/Face-alignment-Trees/raw/master/test_result.png)

The original matlab version can be found in http://www.ics.uci.edu/~xzhu/face/

# Details
I have converted the original model provided by the author to xml format. See 'data/face_p146_small2.xml' for details.

# Dependency
- OpenCV

# Run
For windows, you should use visual studio to build this project. For unix like system, you can use cmake or just write a Makefile to build the project. I don't include the CMakeLists.txt or Makefile in this project because I build it on windows. It should work on unix too. Just take a minute to write a CMakeLists.

# Reference
@inproceedings{zhu2012face,  
  title={Face detection, pose estimation, and landmark localization in the wild},  
  author={Zhu, Xiangxin and Ramanan, Deva},   
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on},  
  pages={2879--2886},  
  year={2012},  
  organization={IEEE}  
}

