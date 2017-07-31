/*********************************************************************/
/*Info:			   C++ implement for facial landmark detection
/*Author:          chen weiliang
/*Version:		   1.1
/*Reference:       face detection pose estimation and landmark localization in the wild
/*Date:			   2015.11.20
/*********************************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>
#include "featpyramid.h"

namespace lm{

	struct Pyramid{
		std::vector<cv::Mat> feat;
		std::vector<float> scales;
		int interval;
		int imx;
		int imy;
		int pady;
		int padx;
	};

	//Model data structure to cache various statistics
	struct ModelComponent{
		int defid;
		int filterid;
		int parent;
		int sizy;
		int sizx;
		int filterI;
		int defI;
		float w[4];
		float scale;
		int starty;
		int startx;
		int step;
		int level;
		cv::Mat_<float> score;
		cv::Mat_<int> Ix;
		cv::Mat_<int> Iy;
	};

	struct Part
	{
		int defid;
		int filterid;
		int parent;
	};

	//Deformable
	struct Def{
		//parameters of the cost function of parts
		float w[4];
		int i;
		//parts未变形时的坐标
		int anchor[3];
	};

	struct Filter{
		int i;
		//5*5的32通道矩阵
		cv::Mat w;
	};

	struct Box{
		float s;
		int c;
		//68*4 或39*4.4列分别为x1,y1,x2,y2
		cv::Mat xy;
		int level;
	};

	bool sortByScore(const Box &b1, const Box &b2);

	class Model
	{
	public:
		Model();
		~Model();

		bool load(std::string modelFile);
		int save(std::string modelFile);
		std::vector<Box> detect(cv::Mat image, uchar options);
		std::vector<Box> detect(cv::Mat image, uchar options, float thresh);
		void setThresh(float thresh);
		float getThresh();
		void setInterval(int interval);
		int getInterval();
		void showResult(cv::Mat &image, std::vector<Box>bs, uchar options);
		std::vector<Box> clipBoxes(cv::Mat src, std::vector<Box>boxes);
		std::vector<int> getAngle(std::vector<Box>bs);
		enum{ LM_SHOW_ANGLE = 1, LM_SHOW_FEATUREPOINT = 2, LM_SHOW_BOXES = 4 };
		enum{
			//Apply Non - maximum suppression.
			LM_DETECT_SUPRESS = 1,
			//Apply boundary checking.
			LM_DETECT_CLIPBOX = 2,
			//Only apply template 8-12
			LM_DETECT_LEFT = 4,
			//Only apply template 0-5
			LM_DETECT_RIGHT = 8,
			//Only apply template 4-8
			LM_DETECT_MIDDLE = 16,
			//Apply all the template.
			LM_DETECT_ALL = 32
		};

	private:
		float mObj;
		float mThresh;
		float mDelta;
		//HOG特征中cell的尺寸，cell的尺寸为sbin*sbin
		int mSbin;
		//HOG金字塔每组的层数
		int mInterval;
		int mLen;
		int mMaxsize[2];
		std::vector<std::vector<Part>> mComponents;
		std::vector<Def> mDefs;
		std::vector<Filter> mFilters;

		//Whether the model is loaded.
		bool mIsLoad;

		void loadComponents(cv::FileStorage fs);
		void loadDefs(cv::FileStorage fs);
		void loadFilters(cv::FileStorage fs);
		void Model::featpyramid(cv::Mat image, Pyramid &pyra);
		void lm::Model::modelcomponents(Pyramid pyra, std::vector<std::vector<lm::ModelComponent>> &components,
			std::vector<cv::Mat>&filters);

		/*
		* shiftdt.cpp
		* Generalized distance transforms based on Felzenswalb and Huttenlocher.
		* This applies computes a min convolution of an arbitrary quadratic function ax^2 + bx
		* This outputs results on an shifted, subsampled grid (useful for passing messages between variables in different domains)
		*/
		void shiftdt(const ModelComponent component, int Nx, int Ny, cv::Mat &msg,
			cv::Mat &Ix, cv::Mat &Iy);
		void dt1d(float *src, float *dst, int *ptr, int step, int len,
			float a, float b, int dshift, int dlen, int dstep);
		// Backtrack through dynamic programming动态规划 messages to estimate part locations
		// and the associated feature vector
		std::vector<cv::Mat> backtrack(cv::Mat x, cv::Mat y,
			std::vector<lm::ModelComponent>parts, Pyramid pyra);
		//Non - maximum suppression.
		//Greedily select high - scoring detections and skip detections
		//that are significantly covered by a previously selected detection.
		std::vector<Box> nmsFace(std::vector<Box>boxes, const float overlap);
		//Find element larger than thresh
		void lm::Model::find(const cv::Mat src, const float thresh, cv::Mat &X, cv::Mat &Y);
	};
}