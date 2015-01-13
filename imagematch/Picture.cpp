#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "Picture.h"

using namespace cv;
using namespace std;

Picture::Picture(char* file)
{
	filename = file;
	loadImage();
	process();
}

Picture::~Picture()
{
	
}

Mat 
Picture::getImg()
{
	return image;
}

vector<KeyPoint>
Picture::getKeyPoints()
{
	return keypoints;
}

Mat 
Picture::getDescriptors()
{
	return descriptors;
}

void
Picture::loadImage(float f)
{
	Mat imageSrc;
	imageSrc = imread(filename);
	if (!imageSrc.data)                       
		cerr << "Could not open or find the image " << filename << std::endl;
	else
		resize(imageSrc, image, Size(), f, f, INTER_AREA);
}

void
Picture::process()
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	featureDetector->detect(image, keypoints, Mat()); // NOTE: featureDetector is a pointer hence the '->'.

	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");
	// Compute the 128 dimension SIFT descriptor at each keypoint.
	// Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
	featureExtractor->compute(image, keypoints, descriptors);
}