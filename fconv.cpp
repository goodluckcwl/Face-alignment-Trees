#include <math.h>
#include <opencv2/opencv.hpp>
#include "featpyramid.h"

using cv::Mat;
using cv::Mat_;
using std::vector;


Mat fconv(cv::Mat feature, cv::Mat filter){
	float *A = (float *)feature.data;
	float *B = (float *)filter.data;
	int A_dims[] = { feature.rows, feature.cols, feature.channels() };
	int B_dims[] = { filter.rows, filter.cols, filter.channels() };
	int C_dims[] = { A_dims[0] - B_dims[0] + 1, A_dims[1] - B_dims[1] + 1, 1 };
	Mat res = Mat(C_dims[0], C_dims[1], CV_32FC(C_dims[2]));
	float *C = (float *)res.data;

	if (A_dims[2] != B_dims[2] || A_dims[0] < B_dims[0] || A_dims[1] < B_dims[1]){
		fprintf(stderr, "Error in fconv:Invalid input.\n");
		return Mat();
	}

	int num_features = A_dims[2];

	int A_step = feature.step1();
	int B_step = filter.step1();					//每行元素个数
	float *dst = (float *)res.data;
	for (int y = 0; y < C_dims[0]; ++y){
		for (int x = 0; x < C_dims[1]; ++x){
			float val = 0;
			float *Axy = A + x*num_features + y*A_step;
			for (int yp = 0; yp < B_dims[0]; ++yp){			//逐行处理
				float *A_off = Axy + yp*A_step;
				float *B_off = B + yp*B_step;
				for (int k = 0; k < B_step; ++k){			//每一行对应相乘
					val += *(A_off++) * *(B_off++);
				}
			}
			*(dst++) = val;
		}
	}

	return res;
}

vector<Mat> fconv(Mat feature, vector<Mat> filters){
	vector<Mat>response;
	for (int i = 0; i < filters.size(); ++i){
		Mat filter = filters[i];
		float *A = (float *)feature.data;
		float *B = (float *)filter.data;
		int A_dims[] = { feature.rows, feature.cols, feature.channels() };
		int B_dims[] = { filter.rows, filter.cols, filter.channels() };
		int C_dims[] = { A_dims[0] - B_dims[0] + 1, A_dims[1] - B_dims[1] + 1 ,1};
		Mat res = Mat(C_dims[0],C_dims[1],CV_32FC(C_dims[2]));

		if (A_dims[2] != B_dims[2] || A_dims[0]<B_dims[0] || A_dims[1]<B_dims[1]){
			fprintf(stderr, "Error in fconv:Invalid input.\n");
			return response;
		}

		int num_features = A_dims[2];

		int A_step = feature.step1();
		int B_step = filter.step1();					//每行元素个数
		float *dst = (float *)res.data;
		for (int y = 0; y < C_dims[0]; ++y){
			for (int x = 0; x < C_dims[1]; ++x){
				float val = 0;
				float *AXY = A + y*A_step + x*num_features;
				for (int yp = 0; yp < B_dims[0]; ++yp){			//逐行处理
					float *A_off = AXY + yp*A_step;
					float *B_off = B + yp*B_step;
					switch (B_step)
					{
					case 20: val += A_off[19] * B_off[19];
					case 19: val += A_off[18] * B_off[18];
					case 18: val += A_off[17] * B_off[17];
					case 17: val += A_off[16] * B_off[16];
					case 16: val += A_off[15] * B_off[15];
					case 15: val += A_off[14] * B_off[14];
					case 14: val += A_off[13] * B_off[13];
					case 13: val += A_off[12] * B_off[12];
					case 12: val += A_off[11] * B_off[11];
					case 11: val += A_off[10] * B_off[10];
					case 10: val += A_off[9] * B_off[9];
					case 9: val += A_off[8] * B_off[8];
					case 8: val += A_off[7] * B_off[7];
					case 7: val += A_off[6] * B_off[6];
					case 6: val += A_off[5] * B_off[5];
					case 5: val += A_off[4] * B_off[4];
					case 4: val += A_off[3] * B_off[3];
					case 3: val += A_off[2] * B_off[2];
					case 2: val += A_off[1] * B_off[1];
					case 1: val += A_off[0] * B_off[0];
						break;
					default:
						for (int k = 0; k < B_step; ++k){			//每一行对应相乘
							val += *(A_off++) * *(B_off++);
						}
					}
				}
				*(dst++) = val;
			}
		}
		response.push_back(res);

		//print
		//std::stringstream matfile;
		//matfile << "E:/score_ " << i << ".xml";
		//saveMat(matfile.str().c_str(), res);
	}

	return response;
}