#include "opencv2/opencv.hpp"
#include "LandmarkModel.h"
#include "time.h"

#define XML_READ
using cv::Mat;
using cv::Mat_;
using cv::Rect;
using cv::FileNode;
using cv::FileStorage;
using cv::FileNodeIterator;
using std::string;
using std::vector;
using cv::Scalar;
using cv::Point;
using namespace lm;

void showBoxes(Mat image, vector<Box>bs, int *posemap){
	Scalar blue = Scalar(255,0,0);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
	for (int i = 0; i < bs.size(); ++i){
		Mat xy = bs[i].xy;
		int minx =image.cols-1;
		int miny =image.rows-1;
		int maxx = 0;
		int maxy = 0;
		for (int j = 0; j < xy.rows; ++j){
			int *s = (int *)xy.data+j*xy.cols;
			int x1 = *s;
			int y1 = *(s + 1);
			int x2 = *(s + 2);
			int y2 = *(s + 3);
			int x = round((x1 + x2) / 2.0);
			int y = round((y1 + y2) / 2.0);
			Rect r = Rect(x1, y1, x2 - x1, y2 - y1);
			rectangle(image, r, blue, 1, 8, 0);
			circle(image, Point(x, y), 1, red, 2, 8, 0);
			minx = MIN(minx, x1);
			miny = MIN(miny, y1);
			maxx = MAX(maxx, x2);
			maxy = MAX(maxy, y2);
		}
		std::stringstream ss;
		ss << posemap[bs[i].c];
		putText(image, ss.str(), Point((minx+maxx)/2, miny), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, green, 1, 8, false);
	}
	imshow("结果", image);
}

int main(int argc, char** argv)
{
	Scalar blue = Scalar(255, 0, 0);
	Scalar green = Scalar(0, 255, 0);
	Scalar red = Scalar(0, 0, 255);
#ifdef XML_READ
//	FileStorage fs("model.xml", cv::FileStorage::READ);
	//FileNode n = fs["components"];
	//for (FileNodeIterator it = n.begin(); it != n.end(); ++it){
	//	FileNode nn = (*it)["component"];

	//	for (FileNodeIterator itt = nn.begin(); itt != nn.end(); ++itt){
	//		int parent; int id;
	//		(*itt)["parent"] >>parent ;
	//		(*itt)["id"] >> id;
	//	}
	//}

	//for (int i = 0; i < 2; ++i){
	//	std::stringstream matfile1, matfile2, matfile3;
	//	matfile1 << "E:/XY_" << i << ".xml";
	//	matfile2 << "E:/m_XY_" << i << ".xml";
	//	matfile3 << "E:/C_Ix_" << i << ".xml";

	//	Mat a, b;
	//	a = readMat(matfile1.str().c_str(), 1);
	//	b = readMat(matfile2.str().c_str(), 1);
	//	Mat c = compare(a, b, 0.00001);
	//	saveMat(matfile3.str().c_str(), c);

	//}

	string a; 

	lm::Model m;
	m.load("data/face_p146_small2.xml");
	m.setInterval(10);
	m.setThresh(MIN(-0.65, m.getThresh()));

	Mat image=cv::imread("F:/image2.jpg");  
	if (!image.data){
		fprintf(stderr, "Error:The input image is invalid.\n");
		return -1;
	}
	double tic = cvGetTickCount();
	vector<Box>bs=m.detect(image,lm::Model::LM_DETECT_SUPRESS|lm::Model::LM_DETECT_ALL);
	double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
	toc = toc / 1000000;
	float fps = 1 / toc;
	std::stringstream ss;
	ss<< "fps"<<fps;
	m.showResult(image, bs,lm::Model::LM_SHOW_ANGLE|lm::Model::LM_SHOW_BOXES|lm::Model::LM_SHOW_FEATUREPOINT);
	putText(image,ss.str(),Point(10,15),cv::FONT_HERSHEY_PLAIN,0.7,red,1,8,false);
	imshow("结果", image);
	cvWaitKey(0);
#else
	Mat m1 = Mat_<int>::ones(3, 3);
	Mat m2 = Mat_<int>::ones(3, 3);
	cv::FileStorage fs("model.xml", cv::FileStorage::WRITE);
	fs << "components" << "[";
	fs << "{";
	fs<<"component"<<"[";
	fs << "{"<< "parent" << 1 << "id" << 2 <<"}";
	fs << "{" << "parent" << 1 << "id" << 2 << "}";
	fs << "{" << "parent" << 1 << "id" << 2 << "}";
	fs << "{" << "parent" << 1 << "id" << 2 << "}";

	
	fs<<"]"<<"}";
	
	fs << "{" << "component" << 7 << "}";
	fs << "{" << "component" << 7 << "}";
	fs << "{" << "component" << 7 << "}";
	fs << "{" << "component" << "["<<m1<<m2<<"]"<< "}";

	fs << "]";
#endif


//	fs.release();
	return 0;
}