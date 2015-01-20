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
Picture::getImgCopy()
{
	return image.clone();
}

float
Picture::diagonal()
{
	Size s = image.size();
	return sqrtf(s.width*s.width + s.height*s.height);
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
	/* try different detectors
	"FAST" – FastFeatureDetector
	"STAR" – StarFeatureDetector
	"SIFT" – SIFT (nonfree module)
	"SURF" – SURF (nonfree module)
	"ORB" – ORB
	"BRISK" – BRISK
	"MSER" – MSER
	"GFTT" – GoodFeaturesToTrackDetector
	"HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
	"Dense" – DenseFeatureDetector
	"SimpleBlob" – SimpleBlobDetector*/
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	featureDetector->detect(image, keypoints, Mat()); 

	/* and try different extractors
	"SIFT" – SIFT
	"SURF" – SURF
	"BRIEF" – BriefDescriptorExtractor
	"BRISK" – BRISK
	"ORB" – ORB
	"FREAK" – FREAK*/
	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");
	featureExtractor->compute(image, keypoints, descriptors);
}

void
Picture::filterMatches(vector<DMatch> &in, vector<DMatch> &out, float ratio)
{
	float distanceSum = 0;
	float min, max;
	bool init = true;
	for (vector<DMatch>::iterator it = in.begin(); it != in.end(); ++it)
	{
		if (init)
		{
			min = max = it->distance;
			init = false;
		}
		else
		{
			min = fmin(min, it->distance);
			max = fmax(min, it->distance);
		}
	}

	float threshold = min + (max - min) * ratio;

	for (vector<DMatch>::iterator it = in.begin(); it != in.end(); ++it)
		if (it->distance < threshold)
			out.push_back(*it);
}

void
Picture::matchTo(Mat trainDescriptors, vector<DMatch> &matches)
{
	vector<DMatch> allMatches;
	Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create("BruteForce");
	descMatcher->match(descriptors, trainDescriptors, allMatches);

	filterMatches(allMatches, matches, 0.1f);
}