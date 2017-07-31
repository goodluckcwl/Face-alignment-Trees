#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "featpyramid.h"

using cv::Mat;
// small value, used to avoid division by zero
#define eps 0.0001
#define	round(x)	((x-floor(x))>0.5 ? ceil(x) : floor(x))


// unit vectors used to compute gradient orientation
double uu[9] = { 1.0000,
0.9397,
0.7660,
0.500,
0.1736,
-0.1736,
-0.5000,
-0.7660,
-0.9397 };
double vv[9] = { 0.0000,
0.3420,
0.6428,
0.8660,
0.9848,
0.9848,
0.8660,
0.6428,
0.3420 };


//输入图像为彩色图像，数据为double类型
void features(Mat image,int sbin,int padx,int pady,Mat &feat_mat){

	if (image.channels() != 3 || image.depth() != CV_32F){
		fprintf(stderr, "Invalid image input.\n");
	}
	const int dims[3] = {image.rows,image.cols,image.channels()};
	// memory for caching orientation histograms & their norms
	int blocks[2];

	//这里用block来表示一个8x8的小块，paper里是用cell来表示，表搞混了，这个数组存的是H和W各可以划分多少cell 
	blocks[0] = (int)round((float)dims[0] / (float)sbin);	//h方向多少个cell  
	blocks[1] = (int)round((float)dims[1] / (float)sbin);	//w方向多少个cell  
	//存梯度的直方图，每个方向一页，总共18个方向，共18页  
	//必须初始化
	float *hist = (float *)calloc(blocks[0] * blocks[1] * 18 , sizeof(float));
	float *norm = (float *)calloc(blocks[0] * blocks[1] , sizeof(float));

	// memory for HOG features
	int out[3];
	out[0] = MAX(blocks[0] - 2, 0);	//减2的原因：因为图像没有扩展，直方图的第一行，第一列和最后一行，最后一列不方便计算，所以宽、高各减去一 
	out[1] = MAX(blocks[1] - 2, 0);
	out[2] = 27 + 4 + 1;			//每个cell的最终输出维度为32维，不同于原始HOG的36维 
	
	//输出的特征,32维对应输出特征矩阵的32个通道
	feat_mat = Mat(out[0] + pady * 2, out[1] + padx * 2, CV_32FC(out[2]));


	cv::Vec<float, 32>init_val;
	init_val = 0;
	init_val[out[2] - 1] = 1;    //write boundary occlusion feature
	feat_mat.setTo(init_val);

	int visible[2];//输入图像不一定是cell大小的整数倍，因此要进行裁剪，这里存的是裁剪后的H,W； 
	visible[0] = blocks[0] * sbin;//h方向
	visible[1] = blocks[1] * sbin;//w方向

	float *im = (float *)image.data;
	int step = image.step1();
	//这个循环计算梯度方向和幅值，并投影到相应的梯度直方图中 
	for (int y = 1; y < visible[0] - 1; y++) {
		float *imgpt = im + MIN(y, dims[0] - 2)*step;
		for (int x = 1; x < visible[1] - 1; x++) {
			// first color channel
			float *s = imgpt + MIN(x, dims[1] - 2)*dims[2];
			float dy = *(s + step) - *(s - step);
			float dx = *(s + dims[2]) - *(s - dims[2]);
			float v = dx*dx + dy*dy;

			// second color channel
			s++;
			float dy2 = *(s + step) - *(s - step);
			float dx2 = *(s + dims[2]) - *(s - dims[2]);
			float v2 = dx2*dx2 + dy2*dy2;

			// third color channel
			s++;
			float dy3 = *(s + step) - *(s - step);
			float dx3 = *(s + dims[2]) - *(s - dims[2]);
			float v3 = dx3*dx3 + dy3*dy3;

			// pick channel with strongest gradient
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			//找到当前的梯度应该投影到哪个方向，[0, 2xPI]总共18个
			// snap to one of 18 orientations
			float best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++) {
				float dot = uu[o] * dx + vv[o] * dy;
				if (dot > best_dot) {
					best_dot = dot;
					best_o = o;
				}
				else if (-dot > best_dot) {
					best_dot = -dot;
					best_o = o + 9;
				}
			}

			//下边这几行代码就是用来线性插值的，注意这里没有使用三线性插值和原始HOG不一样  
			//省略了梯度的插值  
			// add to 4 histograms around pixel using linear interpolation
			float xp = ((float)x + 0.5) / (float)sbin - 0.5;
			float yp = ((float)y + 0.5) / (float)sbin - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			float vx0 = xp - ixp;
			float vy0 = yp - iyp;
			float vx1 = 1.0 - vx0;
			float vy1 = 1.0 - vy0;
			v = sqrt(v);

			float *ttt = hist + ixp*blocks[0] + iyp + best_o*blocks[0] * blocks[1];
			//当前像素对左下角cell有贡献
			if (ixp >= 0 && iyp >= 0) {
				*(hist + ixp*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx1*vy1*v;
			}

			if (ixp + 1 < blocks[1] && iyp >= 0) {
				*(hist + (ixp + 1)*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx0*vy1*v;
			}

			if (ixp >= 0 && iyp + 1 < blocks[0]) {
				*(hist + ixp*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx1*vy0*v;
			}

			if (ixp + 1 < blocks[1] && iyp + 1 < blocks[0]) {
				*(hist + (ixp + 1)*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx0*vy0*v;
			}
		}
	}

	// 因为上边是把[0， 2PI]分为18个方向，举个例子10度和190度算作两个方向  
	// 这里归一化的时候要把10度和190度两个方向算作一个方向，因此要加在一起然后求平方  
	// norm是blocks[0]*blocks[1]大小的，每一个位置存的是所有梯度方向的平方和  
	// compute energy in each block by summing over orientations
	for (int o = 0; o < 9; o++) {
		float *src1 = hist + o*blocks[0] * blocks[1];
		float *src2 = hist + (o + 9)*blocks[0] * blocks[1];
		float *dst = norm;
		float *end = norm + blocks[1] * blocks[0];
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	// compute features
	//计算特征，out[0] = blocks[0] - 2, out[1] = blocks[1] - 2; 防止越界  
	float *feat_data = feat_mat.ptr<float>(padx, pady);
	const int feat_step = feat_mat.step1();					//每行个数
	for (int y = 0; y < out[0]; y++) {
		float *dst = feat_data + y*feat_step;	//逐行处理
		for (int x = 0; x < out[1]; x++) {
			
			float *src, *p, n1, n2, n3, n4;
			//根据上边计算出的energy求出归一化因子  
			//每个cell分属四个block（这个block是2x2个cell的那个block！表混淆）,因此要归一化四次，下边就是求四个归一化因子  
			p = norm + (x + 1)*blocks[0] + y + 1;
			n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + (x + 1)*blocks[0] + y;
			n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y + 1;
			n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);
			p = norm + x*blocks[0] + y;
			n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + eps);

			float t1 = 0;
			float t2 = 0;
			float t3 = 0;
			float t4 = 0;

			// contrast-sensitive features
			//这里把18个方向作为18个特征，也就是10度和190度是不同的特征 
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 18; o++) {
				float h1 = MIN(*src * n1, 0.2);//clip, 大于0.2的特征值截断 
				float h2 = MIN(*src * n2, 0.2);
				float h3 = MIN(*src * n3, 0.2);
				float h4 = MIN(*src * n4, 0.2);
				*(dst++) = 0.5 * (h1 + h2 + h3 + h4);//四个归一化之后的特征值求和除以2
				t1 += h1;//当前cell所在的四个block归一化后的特征值分别加起来
				t2 += h2;
				t3 += h3;
				t4 += h4;
				src += blocks[0] * blocks[1];
			}

			// contrast-insensitive features
			//这里把10度和190度算作一个特征，所以要求一个sum然后再归一化四次  
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 9; o++) {
				float sum = *src + *(src + 9 * blocks[0] * blocks[1]);
				float h1 = MIN(sum * n1, 0.2);
				float h2 = MIN(sum * n2, 0.2);
				float h3 = MIN(sum * n3, 0.2);
				float h4 = MIN(sum * n4, 0.2);
				*(dst++) = 0.5 * (h1 + h2 + h3 + h4);
				src += blocks[0] * blocks[1];
			}

			// texture features
			//纹理特征，cell所在的四个block的特征值的和乘以一个系数
			*(dst++) = 0.2357 * t1;
			*(dst++) = 0.2357 * t2;
			*(dst++) = 0.2357 * t3;
			*(dst++) = 0.2357 * t4;

			// truncation feature
			//最后一个特征是0
			*(dst++) = 0;
		}
	}

	free(hist);
	free(norm);

	return ;
}