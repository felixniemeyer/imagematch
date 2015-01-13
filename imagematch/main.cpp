#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <iostream>
#include <vector>

#include "Picture.h"


using namespace cv;
using namespace std;

void updatePosition(int x, int y)
{

}

void mouseClick(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "clicked at " << x << ", " << y << "\n";
		updatePosition(x,y);
	}
}

int main(int argc, char** argv)
{
	initModule_features2d();
	initModule_nonfree(); 
	
	vector<Picture> pictures;
	pictures.push_back(Picture("./img/IMG_4932.JPG"));
	pictures.push_back(Picture("./img/IMG_4934.JPG"));

	// If you would like to draw the detected keypoint just to check
	Mat keypointvis;
	drawKeypoints(pictures[0].getImg(), pictures[0].getKeyPoints(), keypointvis, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	char
		*windowFrom = "click somewhere at this image",
		*windowTo = "see the same spot in this image";

	namedWindow(windowFrom, CV_WINDOW_AUTOSIZE);
	namedWindow(windowTo, CV_WINDOW_AUTOSIZE);
	setMouseCallback(windowFrom, mouseClick, NULL);

	imshow(windowFrom, keypointvis);                   // Show our image inside it.
	imshow(windowTo, pictures[1].getImg());                   // Show our image inside it.

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}