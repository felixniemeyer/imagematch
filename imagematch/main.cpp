// imagematch
// Felix Niemeyer
// 2015

// we use openCV for keypoint detection, description, matching & for ui drawing
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <iostream>
#include <vector>

#include "Picture.h"
#include "Transformer.h"

using namespace cv;
using namespace std;

// we will use 3 opencv windows
char
*windowFrom = "click somewhere at this image",
*windowTo = "see the same spot in this image",
*windowMatches = "these are common points of the 2 images";

vector<Picture> images; // list of images
int imgIdFrom, imgIdTo; // As we can have multiple images loaded, we have these to variables, which are indexes to the "images" vector
Transformer *transformer = NULL; // transforms position from image A to position in image B

void mouseClick(int event, int x, int y, int flags, void* userdata);
void updatePosition(int x, int y);
void drawMarker(char* window, int imgIndex, int x, int y);

// this initializes the transformer for a pair of images, specified by from- and to-imageIndex
void selectFromTo(int fromImageIndex, int toImageIndex);

int main(int argc, char** argv)
{
	initModule_features2d();
	initModule_nonfree(); 

	if (argc < 3) {
		cout << "pass at least 2 images\n";
	}
	else
	{
		// loading and processing images
		std::cout << "Finding keypoints + matching ...\n";
		for (int i = 1; i < argc; ++i)
			images.push_back(Picture(argv[i]));
	}

	namedWindow(windowMatches, CV_WINDOW_AUTOSIZE);
	namedWindow(windowFrom, CV_WINDOW_AUTOSIZE);
	namedWindow(windowTo, CV_WINDOW_AUTOSIZE);
	
	// initialize the transformer, so we can process clicks. 
	selectFromTo(0, 1);
	setMouseCallback(windowFrom, mouseClick, NULL);

	// exit on any keypress
	waitKey(0);
	return 0;
}

void mouseClick(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
		updatePosition(x, y);
}

void updatePosition(int x, int y)
{
	Transformer::Position<int> guess = transformer->transformPosition(x, y);
	
	cout << "clicked at " << x << ", " << y << "\n";
	cout << "found similar spot at " << guess.x << ", " << guess.y << "\n";

	drawMarker(windowFrom, imgIdFrom, x, y);
	drawMarker(windowTo, imgIdTo, guess.x, guess.y);
}

void drawMarker(char* window, int imgIndex, int x, int y)
{
	Mat img = images[imgIndex].getImgCopy(); // we want to draw on a copy of the original
	rectangle(img, Point(x - 5, y - 5), Point(x + 5, y + 5), Scalar(0, 0, 255), 3, 0);
	imshow(window, img);
}

void selectFromTo(int fromIndex, int toIndex)
{
	// update global variables
	imgIdFrom = fromIndex;
	imgIdTo = toIndex;

	// retrieve matches
	vector<DMatch> matches;
	images[fromIndex].matchTo(images[toIndex].getDescriptors(), matches);

	// initialize transformer
	if (transformer != NULL)
		delete transformer;
	transformer = new Transformer(matches, images[fromIndex].getKeyPoints(), images[toIndex].getKeyPoints(), images[fromIndex].diagonal());

	// Draw keypoints and matches
	Mat kpFrom, kpTo, matchesImg; //images
	drawKeypoints(images[fromIndex].getImgCopy(), images[fromIndex].getKeyPoints(), kpFrom, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(images[toIndex].getImgCopy(), images[toIndex].getKeyPoints(), kpTo, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawMatches(
		images[fromIndex].getImgCopy(), images[fromIndex].getKeyPoints(),
		images[toIndex].getImgCopy(), images[toIndex].getKeyPoints(),
		matches,
		matchesImg);

	imshow(windowFrom, kpFrom);
	imshow(windowTo, kpTo);
	imshow(windowMatches, matchesImg);
}