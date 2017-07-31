/*********************************************************************/
/*Info:			   C++ implement for facial landmark detection
/*Author:		   chen weiliang
/*Version:		   1.1.
/*Reference:       face detection pose estimation and landmark localization in the wild
/*Date:			   2015.11.20
/*********************************************************************/
#include "LandmarkModel.h"
#include "opencv2/opencv.hpp"
#include <math.h>
//#include <chrono>//system clock
#include <random>
#include <algorithm>
#include "Utils.h"

#define INF 1E20
static inline int square(int x) { return x*x; }

using cv::Mat_;
using cv::Mat;
using cv::FileStorage;
using cv::FileNode;
using cv::FileNodeIterator;
using std::string;
using std::vector;
using cv::Scalar;
using cv::Point;
using cv::Rect;

//*******************************Class Model*******************************
//Keep track of detected boxes and features
int const BOXCACHESIZE = 100000;

const int POSEMAP[] = { 90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90 };

lm::Model::Model()
{
	mIsLoad = false;
}


lm::Model::~Model()
{
}

bool lm::Model::load(string modelFile){
	FileStorage fs;
	if (!fs.open(modelFile, FileStorage::READ)){
		fprintf(stderr, "The model file cannot be open.\n");
		return -1;
	}

	fs["obj"] >> mObj;
	fs["thresh"] >> mThresh;
	fs["delta"] >> mDelta;
	fs["sbin"] >> mSbin;
	fs["interval"] >> mInterval;
	fs["len"] >> mLen;

	vector<int>maxsizeval;
	fs["maxsize"] >> maxsizeval;
	mMaxsize[0] = maxsizeval[0];
	mMaxsize[1] = maxsizeval[1];

	//components
	loadComponents(fs);

	//defs
	loadDefs(fs);

	//filters
	loadFilters(fs);

	fs.release();
	mIsLoad = true;
	return true;
}

void lm::Model::loadComponents(FileStorage fs){

	FileNode fn = fs["components"];
	for (FileNodeIterator it = fn.begin(); it != fn.end(); ++it){
		FileNode fnn = (*it)["component"];
		vector<Part> com;
		for (FileNodeIterator itt = fnn.begin(); itt != fnn.end(); ++itt){
			Part p;
			(*itt)["defid"] >> p.defid;
			(*itt)["filterid"] >> p.filterid;
			(*itt)["parent"] >> p.parent;
			com.push_back(p);
		}
		mComponents.push_back(com);
	}
}

void lm::Model::loadFilters(FileStorage fs){
	FileNode fn = fs["filters"];
	for (FileNodeIterator it = fn.begin(); it != fn.end(); ++it){
		Filter f;
		(*it)["i"] >> f.i;
		(*it)["w"] >> f.w;
		int channels = 32;
		f.w = f.w.reshape(channels, f.w.rows);

		mFilters.push_back(f);
	}

}

void lm::Model::loadDefs(FileStorage fs){
	FileNode fn = fs["defs"];
	for (FileNodeIterator it = fn.begin(); it != fn.end(); ++it){
		Def def;
		vector<float>w;
		(*it)["w"] >> w;
		def.w[0] = w[0];
		def.w[1] = w[1];
		def.w[2] = w[2];
		def.w[3] = w[3];
		(*it)["i"] >> def.i;
		vector<int>anchor;
		(*it)["anchor"] >> anchor;
		def.anchor[0] = anchor[0];
		def.anchor[1] = anchor[1];
		def.anchor[2] = anchor[2];
		mDefs.push_back(def);
	}
}

int lm::Model::save(string modelFile){
	FileStorage fs;
	if (!fs.open(modelFile, FileStorage::WRITE)){
		fprintf(stderr, "The model file cannot be open.\n");
	}

	fs << "obj" << mObj;
	fs << "thresh" << mThresh;
	fs << "delta" << mDelta;
	fs << "sbin" << mSbin;
	fs << "interval" << mInterval;
	fs << "len" << mLen;

	fs << "maxsize" << "[:" << mMaxsize[0] << mMaxsize[1] << "]";

	fs << "components" << "[";
	for (vector<vector<Part>>::iterator it = mComponents.begin(); it != mComponents.end(); ++it){
		vector<Part> com = *it;
		fs << "{" << "component";
		fs << "[";
		for (vector<Part>::iterator itt = com.begin(); itt != com.end(); ++itt){
			fs << "{";
			fs << "defid" << (*itt).defid;
			fs << "filterid" << (*itt).filterid;
			fs << "parent" << (*itt).parent;
			fs << "}";
		}
		fs << "]" << "}";
	}
	fs << "]";

	fs << "defs" << "[";
	for (vector<Def>::iterator it = mDefs.begin(); it != mDefs.end(); ++it){
		fs << "{";
		fs << "w" << "[" << (*it).w[0] << (*it).w[1];
		fs << (*it).w[2] << (*it).w[3] << "]";
		fs << "i" << (*it).i;
		fs << "anchor" << "[" << (*it).anchor[0] << (*it).anchor[1] << (*it).anchor[2] << "]";
		fs << "}";
	}
	fs << "]";

	fs << "filters" << "[";
	for (vector<Filter>::iterator it = mFilters.begin(); it != mFilters.end(); ++it){
		fs << "{";
		fs << "i" << (*it).i;
		Mat t_w = Mat((*it).w.rows, (*it).w.cols*(*it).w.channels(), CV_32F, (*it).w.data);
		fs << "w" << t_w;

		fs << "}";
	}
	fs << "]";

	fs.release();
	return 0;
}

//Detect facial landmark.
//image:	The input color image.
//options:  Options for detection.It can be any combination of LM_DETECT_SUPRESS,
//          LM_DETECT_CLIPBOX,LM_DETECT_LEFT,LM_DETECT_RIGHT,LM_DETECT_MIDDLE,LM_DETECT_ALL.
vector<lm::Box> lm::Model::detect(Mat image, uchar options){
	if (!mIsLoad){
		fprintf(stderr, "Error using detect:The model has not bean loaded.\n");
		return vector<Box>();
	}

	if (image.channels() != 3 || !image.data){
		fprintf(stderr, "Error using detect:Invalid input.\n");
		return vector<Box>();
	}

	int cnt = 0;

	Box initBox;
	initBox.c = 0;
	initBox.s = 0;
	initBox.level = 0;
	vector<lm::Box>boxes(BOXCACHESIZE, initBox);

	//Compute the feature pyramid and prepare filters
	Pyramid pyra;
	double tic = cvGetTickCount();
	featpyramid(image, pyra);
	double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
	toc = toc / 1000000;
	fprintf(stderr, "featPyramind:%f s\n", toc);

	//Cache various statistics from the model data structure for later use
	vector<Mat>filters;
	vector<vector<ModelComponent>> components;
	modelcomponents(pyra, components, filters);

	//Cache 
	vector<vector<Mat>>resp;
	resp.resize(pyra.feat.size());

	vector<int>c;
	if (options & LM_DETECT_LEFT){
		c.push_back(8);
		c.push_back(9);
		c.push_back(10);
		c.push_back(11);
		c.push_back(12);
	}
	else if (options & LM_DETECT_RIGHT){
		c.push_back(0);
		c.push_back(1);
		c.push_back(2);
		c.push_back(3);
		c.push_back(4);
	}
	else if (options & LM_DETECT_MIDDLE){
		c.push_back(4);
		c.push_back(5);
		c.push_back(6);
		c.push_back(7);
		c.push_back(8);
	}
	else{
		for (int i = 0; i < mComponents.size(); i++){
			c.push_back(i);
		}
	}

	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::shuffle(c.begin(), c.end(), std::default_random_engine(seed));

	//For each template
	for (int i = 0; i < c.size(); ++i){
		int minLevel = mInterval + 1;

		vector<int>levels;
		for (int j = minLevel - 1; j < pyra.feat.size() - 1; ++j){
			levels.push_back(j);
		}
		//unsigned s = std::chrono::system_clock::now().time_since_epoch().count();
		//std::shuffle(levels.begin(), levels.end(), std::default_random_engine(s));

		for (int k = 0; k < levels.size(); ++k){
			vector<ModelComponent>parts = components[c[i]];
			int numParts = parts.size();
			//Local part scores
			for (int kk = 0; kk < numParts; ++kk){
				int f = parts[kk].filterid;
				int level = levels[k] - parts[kk].scale*mInterval;
				//若已缓存，则无需重复计算
				if (!resp[level].size()){
					double tic = cvGetTickCount();
					resp[level] = fconv(pyra.feat[level], filters);
					double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
					toc = toc / 1000000;
					fprintf(stderr, "fconv %d time:%f s\n", level, toc);
				}
				//必须复制
				resp[level][f - 1].copyTo(parts[kk].score);
				parts[kk].level = level;
			}

			// Walk from leaves to root of tree, passing message to parent
			// Given a 2D array of filter scores 'child', shiftdt() does the following :
			// (1) Apply distance transform
			// (2) Shift by anchor position(child.startxy) of part wrt parent
			// (3) Downsample by child.step
			for (int kk = numParts - 1; kk>0; --kk){
				ModelComponent child = parts[kk];
				int par = child.parent;
				int Ny = parts[par - 1].score.rows;
				int Nx = parts[par - 1].score.cols;
				Mat msg;
				shiftdt(parts[kk], Nx, Ny, msg, parts[kk].Ix, parts[kk].Iy);
				parts[par - 1].score = parts[par - 1].score + msg;
			}
			// Add bias to root score
			Mat rscore = parts[0].score + parts[0].w[0];

			//找到所需要的元素的位置
			Mat X, Y;
			find(rscore, mThresh, X, Y);

			//每个都是68*4或39*4的大小。4列分别为x1,y1,x2,y2.
			vector<Mat>XY;
			if (X.cols){
				XY = backtrack(X, Y, parts, pyra);
			}

			int *pX = (int *)X.data;
			int *pY = (int *)Y.data;
			float *rscore_data = (float *)rscore.data;
			int step = rscore.step1();
			//Walk back down tree following pointers
			for (int m = 0; m < X.cols; ++m){
				int x = *(pX + m);
				int y = *(pY + m);
				if (cnt == BOXCACHESIZE){
					//Greedily select high - scoring detections and skip detections that are 
					//significantly covered by a previously selected detection.
					vector<Box> b0 = nmsFace(boxes, 0.3);
					boxes.clear();
					boxes = vector<Box>(BOXCACHESIZE, initBox);
					cnt = b0.size();
					for (int i = 0; i < cnt; ++i){
						boxes[i] = b0[i];
					}
				}

				boxes[cnt].c = c[i];
				boxes[cnt].s = *(rscore_data + x + y*step);
				boxes[cnt].level = levels[k];
				boxes[cnt].xy = XY[m];
				cnt++;
			}
		}
	}

	vector<lm::Box>r(boxes.begin(), boxes.begin() + cnt);

	if (options & LM_DETECT_SUPRESS){
		r = nmsFace(r, 0.3);
	}
	if (options & LM_DETECT_CLIPBOX){
		r = clipBoxes(image, r);
	}

	return r;
}

//Detect facial landmark with thresh.
//image:	The input color image.
//options:  Options for detection.It can be any combination of LM_DETECT_SUPRESS,
//          LM_DETECT_CLIPBOX,LM_DETECT_LEFT,LM_DETECT_RIGHT,LM_DETECT_MIDDLE,LM_DETECT_ALL.
//thresh:	-0.65 is appropriate.
vector<lm::Box> lm::Model::detect(Mat image, uchar options, float thresh){
	mThresh = thresh;
	return detect(image, options);
}

void lm::Model::featpyramid(Mat image, Pyramid &pyra){
	//Select padding, allowing for one cell in model to be visible
	//Even padding allows for consistent spatial relations across 2X scales
	int padx = MAX(mMaxsize[1] - 1 - 1, 0);
	int pady = MAX(mMaxsize[0] - 1 - 1, 0);
	//padx = model.maxsize[1];
	//pady = model.maxsize[2];
	//padx = ceil(padx / 2) * 2;
	//pady = ceil(pady / 2) * 2;

	float sc = pow(2, 1.0 / mInterval);
	int max_scale = 1 + floor(log(MIN(image.rows, image.cols) / (5 * mSbin)) / log(sc));

	//
	pyra.feat.resize(max_scale + mInterval);
	pyra.scales.resize(max_scale + mInterval);

	//
	Mat image1 = Mat(image.rows, image.cols, CV_32FC(image.channels()));
	image.convertTo(image1, CV_32FC(image.channels()));

	for (int i = 0; i < mInterval; ++i){
		float scale_factor = 1.f / pow(sc, i);
		cv::Size size = cv::Size(round(image1.cols*scale_factor), round(image1.rows*scale_factor));
		Mat scaled;
		resize(image1, scaled, size, 0, 0, CV_INTER_LINEAR);
		//feat[0~mInterval-1] is not used in function detect().
		//pyra.feat[i] = features2(scaled, mSbin / 2, padx + 1, pady + 1);
		pyra.scales[i] = 2 / pow(sc, i);
		//"second" 2x interval
		pyra.feat[i + mInterval] = features2(scaled, mSbin, padx + 1, pady + 1);
		//features(scaled, mSbin, padx + 1, pady + 1, pyra.feat[i + mInterval]);
		pyra.scales[i + mInterval] = 1 / pow(sc, i);
		//remaining interals
		for (int j = i + mInterval; j < max_scale; j += mInterval){
			cv::Size s = cv::Size((int)round(scaled.cols*0.5), (int)round(scaled.rows*0.5));
			Mat reduced = Mat(s, CV_32FC(scaled.channels()));
			resize(scaled, reduced, s, 0, 0, CV_INTER_LINEAR);
			pyra.feat[j + mInterval] = features2(reduced, mSbin, padx + 1, pady + 1);
			//features(reduced, mSbin, padx + 1, pady + 1, pyra.feat[j + mInterval]);
			pyra.scales[j + mInterval] = 0.5*pyra.scales[j];
			scaled = reduced;
		}
	}
	for (int i = 0; i < pyra.scales.size(); ++i){
		pyra.scales[i] = mSbin / pyra.scales[i];
	}
	pyra.interval = mInterval;
	pyra.imy = image.rows;
	pyra.imx = image.cols;
	pyra.padx = padx;
	pyra.pady = pady;
}

void lm::Model::modelcomponents(Pyramid pyra, vector<vector<lm::ModelComponent>> &components,
	vector<Mat>&filters){
	components.resize(mComponents.size());
	for (int i = 0; i < components.size(); ++i){
		vector<ModelComponent>coms;
		coms.resize(mComponents[i].size());
		for (int j = 0; j < mComponents[i].size(); ++j){
			ModelComponent com;
			Part p = mComponents[i][j];
			Filter x = mFilters[p.filterid - 1];
			com.parent = p.parent;
			com.filterid = p.filterid;
			com.defid = p.defid;
			com.sizy = x.w.rows;
			com.sizx = x.w.cols;
			com.filterI = x.i;
			Def def = mDefs[p.defid - 1];
			com.defI = def.i;
			com.w[0] = def.w[0];
			com.w[1] = def.w[1];
			com.w[2] = def.w[2];
			com.w[3] = def.w[3];

			//store the scale of each part relative to the component root
			int par = p.parent;
			assert(par - 1< j);
			int ax = def.anchor[0];
			int ay = def.anchor[1];
			int ds = def.anchor[2];
			if (par>0){
				com.scale = ds + coms[par - 1].scale;
			}
			else{
				assert(j == 0);
				com.scale = 0;
			}
			// amount of(virtual) padding to hallucinate
			int step = pow(2, ds);
			int virtpady = (step - 1)*pyra.pady;
			int virtpadx = (step - 1)*pyra.padx;
			//starting points(simulates additional padding at finer scales)
			com.starty = ay - virtpady;
			com.startx = ax - virtpadx;
			com.step = step;
			com.level = 0;
			//com.score = 0;
			//com.Ix = 0;
			//com.Iy = 0;
			coms[j] = com;
		}
		components[i] = coms;
	}
	filters.resize(mFilters.size());
	for (int i = 0; i < filters.size(); ++i){
		filters[i] = mFilters[i].w;
	}
}

void lm::Model::dt1d(float *src, float *dst, int *ptr, int step, int len,
	float a, float b, int dshift, int dlen, int dstep){
	int *v = new int[len];
	float *z = new float[len + 1];
	int k = 0;
	int q = 0;
	v[0] = 0;
	z[0] = -INF;
	z[1] = +INF;

	for (q = 1; q <= len - 1; q++) {
		float s = ((src[q*step] - src[v[k] * step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2 * a*(q - v[k]));
		while (s <= z[k] && k>0) {
			k--;
			s = ((src[q*step] - src[v[k] * step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2 * a*(q - v[k]));
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = +INF;
	}

	k = 0;
	q = dshift;

	for (int i = 0; i <= dlen - 1; i++) {
		while (z[k + 1] < q)
			k++;
		dst[i*step] = a*square(q - v[k]) + b*(q - v[k]) + src[v[k] * step];
		ptr[i*step] = v[k];
		q += dstep;
	}

	delete[] v;
	delete[] z;
}

//输出坐标Ix,Iy为C风格，即从0开始
void lm::Model::shiftdt(const lm::ModelComponent component, int Nx, int Ny, Mat &msg,
	Mat &Ix, Mat &Iy){
	Mat score = component.score;
	int sizx = score.cols;
	int sizy = score.rows;
	float ax = -component.w[0];
	float bx = -component.w[1];
	float ay = -component.w[2];
	float by = -component.w[3];
	int offx = component.startx - 1;
	int offy = component.starty - 1;
	float step = component.step;
	msg = Mat(Ny, Nx, CV_32F);
	Ix = Mat(Ny, Nx, CV_32S);
	Iy = Mat(Ny, Nx, CV_32S);
	float *msg_data = (float *)msg.data;
	int *Ix_data = (int *)Ix.data;
	int *Iy_data = (int *)Iy.data;

	float *temM = (float *)malloc(Ny*sizx*sizeof(float));
	int *temIy = (int *)malloc(Ny*sizx*sizeof(int));

	for (int x = 0; x < sizx; ++x){			//逐列處理
		dt1d((float *)score.data + x, temM + x, temIy + x, Nx, sizy,
			ay, by, offy, Ny, step);
	}

	for (int y = 0; y < Ny; ++y){			//逐行處理
		dt1d(temM + y*sizx, msg_data + y*Nx, Ix_data + y*Nx, 1, sizx,
			ax, bx, offx, Nx, step);
	}

	//get argmins.
	for (int y = 0; y < Ny; ++y) {
		for (int x = 0; x < Nx; ++x) {
			int p = x + y*Nx;
			Iy_data[p] = temIy[Ix_data[p] + y*Nx];
			//Ix_data[p] = Ix_data[p]+1;
		}
	}

	free(temM);
	free(temIy);
	return;
}

vector<Mat> lm::Model::backtrack(Mat X, Mat Y,
	vector<lm::ModelComponent>parts, Pyramid pyra){
	int numParts = parts.size();
	int cols = X.cols;

	Mat Xptr = Mat(numParts, cols, CV_32S, cv::Scalar(0));
	Mat	Yptr = Mat(numParts, cols, CV_32S, cv::Scalar(0));

	//tmp_box有4个Mat，分别是x1,y1,y2,y2
	//每个Mat是numParts*cols
	vector<Mat>tmp_XY(4);
	for (int i = 0; i < tmp_XY.size(); ++i){
		tmp_XY[i] = Mat(numParts, cols, CV_32S);
	}

	int k = 0;
	ModelComponent p = parts[k];
	//必须复制
	X.copyTo(Xptr.row(k));
	Y.copyTo(Yptr.row(k));


	//image coordinates of root
	float scale = pyra.scales[p.level];
	int padx = pyra.padx;
	int pady = pyra.pady;
	//tmp_box[0].row(k)=(X - 1 - padx)*scale + 1
	tmp_XY[0].row(k) = (X - padx)*scale;
	//tmp_box[1].row(k)=(Y - 1 - pady)*scale + 1
	tmp_XY[1].row(k) = (Y - pady)*scale;
	//tmp_box[2].row(k)=tmp_box[0].row(k) + p.sizx*scale - 1
	tmp_XY[2].row(k) = tmp_XY[0].row(k) + p.sizx*scale - 1;
	//tmp_box[3].row(k)=tmp_box[1].row(k) + p.sizy*scale - 1
	tmp_XY[3].row(k) = tmp_XY[1].row(k) + p.sizy*scale - 1;

	int *xps = (int *)Xptr.data;
	int *yps = (int *)Yptr.data;
	int step = Xptr.step1();
	for (k = 1; k < numParts; ++k){
		p = parts[k];
		int par = p.parent;
		//从Ix，Iy中取出坐标为(X,Y)的点，存入Xptr,Yptr的第k行。

		int *xs = xps + (par - 1)*step;
		int *ys = yps + (par - 1)*step;
		int *Ixs = (int *)p.Ix.data;
		int *Iys = (int *)p.Iy.data;
		int *xpd = xps + k*step;
		int *ypd = yps + k*step;
		for (int i = 0; i < Xptr.cols; ++i){
			int x = *(xs++);
			int y = *(ys++);
			int ind = x + y *(p.Ix.cols);
			*(Iys + ind);
			*(xpd++) = *(Ixs + ind);
			*(ypd++) = *(Iys + ind);
		}
		//第k个part在图像上的坐标
		scale = pyra.scales[p.level];
		tmp_XY[0].row(k) = (Xptr.row(k) - padx)*scale;
		tmp_XY[1].row(k) = (Yptr.row(k) - pady)*scale;
		tmp_XY[2].row(k) = tmp_XY[0].row(k) + p.sizx*scale - 1;
		tmp_XY[3].row(k) = tmp_XY[1].row(k) + p.sizy*scale - 1;
	}

	//重新排列,从tmp_box[0]、tmp_box[1]、tmp_box[2]、tmp_box[3]
	//中各取第i列组成box[i]:numparts*4
	//box有length个mat，每个都是numparts*4的大小.4列分别为x1,y1,x2,y2
	vector<Mat>XY(cols);

	for (int i = 0; i < cols; ++i){
		XY[i] = Mat(numParts, 4, CV_32S);
		int *d = (int *)XY[i].data;
		int *s0 = (int *)tmp_XY[0].data + i;//取第i列
		int *s1 = (int *)tmp_XY[1].data + i;
		int *s2 = (int *)tmp_XY[2].data + i;
		int *s3 = (int *)tmp_XY[3].data + i;
		//每一列
		for (int j = 0; j < numParts; ++j){
			*(d++) = *s0;
			*(d++) = *s1;
			*(d++) = *s2;
			*(d++) = *s3;
			s0 += cols;
			s1 += cols;
			s2 += cols;
			s3 += cols;
		}
	}

	return XY;
}


vector<lm::Box> lm::Model::nmsFace(vector<Box>boxes, const float overlap){
	int N = boxes.size();
	vector<Box>top;
	if (boxes.empty()){
		return top;
	}
	else{
		int numpart = boxes[0].xy.rows;

		//按照score升序排列
		sort(boxes.begin(), boxes.end(), sortByScore);

		//throw away boxes with low score if there are too many candidates
		if (N > 30000){
			boxes = vector<Box>(boxes.end() - 30000, boxes.end());
		}
		N = MIN(30000, N);

		int *x1 = (int *)calloc(N, sizeof(int));
		int *y1 = (int *)calloc(N, sizeof(int));
		int *x2 = (int *)calloc(N, sizeof(int));
		int *y2 = (int *)calloc(N, sizeof(int));
		int *area = (int *)calloc(N, sizeof(int));

		//计算每个人脸的大小
		for (int nb = 0; nb<N; ++nb){
			Mat xy = boxes[nb].xy;
			int *xys = (int *)xy.data;
			if (!xy.isContinuous()){
				fprintf(stderr, "Error using nmsFace:Mat is not continuous.\n");
			}
			if (numpart == 1){
				x1[nb] = *(xys++);
				y1[nb] = *(xys++);
				x2[nb] = *(xys++);
				y2[nb] = *(xys++);
			}
			else{
				x1[nb] = *(xys++);
				y1[nb] = *(xys++);
				x2[nb] = *(xys++);
				y2[nb] = *(xys++);
				for (int i = 1; i < xy.rows; ++i){
					x1[nb] = MIN(x1[nb], *(xys));
					xys++;
					y1[nb] = MIN(y1[nb], *(xys));
					xys++;
					x2[nb] = MAX(x2[nb], *(xys));
					xys++;
					y2[nb] = MAX(y2[nb], *(xys));
					xys++;
				}
			}
			area[nb] = (x2[nb] - x1[nb] + 1)*(y2[nb] - y1[nb] + 1);
		}

		vector<int>I;
		for (int i = 0; i<N; ++i){
			I.push_back(i);
		}

		while (!I.empty()){
			int last = I.back();
			top.push_back(boxes[last]);

			vector<int>suppress;


			//
			int len = I.size() - 1;
			int *xx1 = (int *)malloc(len*sizeof(int));
			int *yy1 = (int *)malloc(len*sizeof(int));
			int *xx2 = (int *)malloc(len*sizeof(int));
			int *yy2 = (int *)malloc(len*sizeof(int));
			int *w = (int *)malloc(len*sizeof(int));
			int *h = (int *)malloc(len*sizeof(int));
			int *inter = (int *)malloc(len*sizeof(int));
			float *o1 = (float *)malloc(len*sizeof(float));
			float *o2 = (float *)malloc(len*sizeof(float));
			for (int j = 0; j < len; ++j){
				int ind = I[j];
				xx1[j] = MAX(x1[last], x1[ind]);
				yy1[j] = MAX(y1[last], y1[ind]);
				xx2[j] = MIN(x2[last], x2[ind]);
				yy2[j] = MIN(y2[last], y2[ind]);
				w[j] = xx2[j] - xx1[j] + 1 > 0 ? xx2[j] - xx1[j] + 1 : 0;
				h[j] = yy2[j] - yy1[j] + 1 > 0 ? yy2[j] - yy1[j] + 1 : 0;
				inter[j] = w[j] * h[j];
				o1[j] = inter[j] * 1.0 / area[ind];
				o2[j] = inter[j] * 1.0 / area[last];
				if (o1[j] > overlap || o2[j] > overlap){
					suppress.push_back(I[j]);
				}
			}
			suppress.push_back(last);

			//从I中删除与suppress相同的元素
			vector<int>tmpI;
			int s = 0;
			for (int j = 0; j < I.size(); ++j){
				if (I[j] != suppress[s]){
					tmpI.push_back(I[j]);
				}
				else{
					s++;
				}
			}
			I = tmpI;
			//
			free(xx1);
			free(yy1);
			free(xx2);
			free(yy2);
			free(w);
			free(h);
			free(inter);
			free(o1);
			free(o2);
		}

		//Free memory
		free(x1);
		free(y1);
		free(x2);
		free(y2);
		free(area);

	}
	return top;
}

//Draw result on the input image.
//image:	The input image.
//bs:		Boxes detected.
//options:	Options for drawing.It can be any combination of LM_SHOW_ANGLE
//			LM_SHOW_BOXES,LM_SHOW_FEATUREPOINT.
void lm::Model::showResult(cv::Mat &image, vector<Box>bs, uchar options){
	Scalar blue = Scalar(255, 0, 0);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
	for (int i = 0; i < bs.size(); ++i){
		Mat xy = bs[i].xy;
		int minx = image.cols - 1;
		int miny = image.rows - 1;
		int maxx = 0;
		int maxy = 0;
		for (int j = 0; j < xy.rows; ++j){
			int *s = (int *)xy.data + j*xy.cols;
			int x1 = *s;
			int y1 = *(s + 1);
			int x2 = *(s + 2);
			int y2 = *(s + 3);
			int x = round((x1 + x2) / 2.0);
			int y = round((y1 + y2) / 2.0);
			Rect r = Rect(x1, y1, x2 - x1, y2 - y1);
			if (options & 0x02){
				circle(image, Point(x, y), 1, red, 2, 8, 0);
			}
			if (options & 0x04){
				rectangle(image, r, blue, 1, 8, 0);
			}
			minx = MIN(minx, x1);
			miny = MIN(miny, y1);
			maxx = MAX(maxx, x2);
			maxy = MAX(maxy, y2);
		}
		std::stringstream ss;
		ss << POSEMAP[bs[i].c];
		if (options & 0x01){
			putText(image, ss.str(), Point((minx + maxx) / 2, miny), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, green, 1, 8, false);
		}
	}
}

//Get angles.
//bs:	Boxes detected.
vector<int> lm::Model::getAngle(std::vector<Box>bs){
	vector<int>angles;
	for (int i = 0; i < bs.size(); ++i){
		angles.push_back(POSEMAP[bs[i].c]);
	}
	return angles;
}

//Find elements larger than thresh.
void lm::Model::find(const cv::Mat src, const float thresh, cv::Mat &X, cv::Mat &Y){
	if (src.depth() != CV_32F || src.channels() != 1){
		fprintf(stderr, "Error in find:Invalid input.\n");
	}
	int h = src.rows;
	int w = src.cols;
	X = Mat(1, h*w, CV_32S);
	Y = Mat(1, h*w, CV_32S);
	float *s = (float *)src.data;
	int *px = (int *)X.data;
	int *py = (int *)Y.data;
	int num = 0;
	for (int y = 0; y < h; ++y){
		for (int x = 0; x < w; ++x){
			if (*(s++)>thresh){
				num++;
				*(px++) = x;
				*(py++) = y;
			}
		}
	}

	//调整大小
	X = X.colRange(0, num);
	Y = Y.colRange(0, num);
}

//Clip boxes to image boundary
vector<lm::Box> lm::Model::clipBoxes(cv::Mat image, std::vector<Box>boxes){
	for (int i = 0; i < boxes.size(); ++i){
		if (boxes[i].xy.depth() != CV_32S || boxes[i].xy.cols != 4){
			fprintf(stderr, "Error using clipBoxes:Invalid input.\n");
			return boxes;
		}
		int *s = (int *)boxes[i].xy.data;
		for (int j = 0; j < boxes[i].xy.rows; ++j){
			*s = MAX(*s, 0);
			s++;
			*s = MAX(*s, 0);
			s++;
			*s = MIN(*s, image.cols - 1);
			s++;
			*s = MIN(*s, image.rows - 1);
			s++;
		}
	}
	return boxes;
}

void lm::Model::setInterval(int interval){
	mInterval = interval;
}

void lm::Model::setThresh(float thresh){
	mThresh = thresh;
}

int lm::Model::getInterval(){
	return mInterval;
}

float lm::Model::getThresh(){
	return mThresh;
}

//******************************Class Model END********************************


bool lm::sortByScore(const Box &b1, const Box &b2){
	return b1.s < b2.s;
}


