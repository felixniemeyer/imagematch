//Felix Niemeyer 2015

//we use openCV for keypoint detection, drawing
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

char
*windowFrom = "click somewhere at this image",
*windowTo = "see the same spot in this image",
*windowMatches = "these are common points of the 2 images";

vector<Picture> pictures;
Transformer *transformer = NULL;
int imgIdFrom, imgIdTo;



void drawMarker(char* window, int imgIndex, int x, int y)
{
	Mat img = pictures[imgIndex].getImgCopy();
	rectangle(img, Point(x - 5, y - 5), Point(x + 5, y + 5), Scalar(0, 0, 255), 3, 0);
	imshow(window, img);
}

void updatePosition(int x, int y)
{
	Transformer::Position<int> guess = transformer->transformPosition(x, y);
	cout << "clicked at " << x << ", " << y << "\n";
	cout << "found similar spot at " << guess.x << ", " << guess.y << "\n";
	drawMarker(windowFrom, imgIdFrom, x, y);
	drawMarker(windowTo, imgIdTo, guess.x, guess.y);
}

void selectFromTo(int fromIndex, int toIndex)
{
	imgIdFrom = fromIndex;
	imgIdTo = toIndex;

	vector<DMatch> matches;
	pictures[fromIndex].matchTo(pictures[toIndex].getDescriptors(), matches);

	if (transformer != NULL)
		delete transformer;

	transformer = new Transformer(matches, pictures[fromIndex].getKeyPoints(), pictures[toIndex].getKeyPoints(), pictures[fromIndex].diagonal());

	// If you would like to draw the detected keypoint just to check
	Mat kpFrom, kpTo; //images
	drawKeypoints(pictures[fromIndex].getImgCopy(), pictures[fromIndex].getKeyPoints(), kpFrom, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(pictures[toIndex].getImgCopy(), pictures[toIndex].getKeyPoints(), kpTo, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	Mat matchesImg;
	drawMatches(
		pictures[fromIndex].getImgCopy(), pictures[fromIndex].getKeyPoints(),
		pictures[toIndex].getImgCopy(), pictures[toIndex].getKeyPoints(),
		matches,
		matchesImg);

	imshow(windowFrom, kpFrom);                   // Show our image inside it.
	imshow(windowTo, kpTo);                   // Show our image inside it.
	imshow(windowMatches, matchesImg);
}

void mouseClick(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		updatePosition(x,y);
	}
}

int main(int argc, char** argv)
{
	initModule_features2d();
	initModule_nonfree(); 

	std::cout << "Finding keypoints + matching ...\n";
	
	pictures.push_back(Picture("./img/IMG_4932.JPG"));
	pictures.push_back(Picture("./img/IMG_4934.JPG"));

	namedWindow(windowMatches, CV_WINDOW_AUTOSIZE);
	namedWindow(windowFrom, CV_WINDOW_AUTOSIZE);
	namedWindow(windowTo, CV_WINDOW_AUTOSIZE);

	selectFromTo(0, 1);

	setMouseCallback(windowFrom, mouseClick, NULL);

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}