#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

class Picture
{
public:
	Picture(char* file);
	~Picture();

	cv::Mat getImg();
	std::vector<cv::KeyPoint> getKeyPoints();
	cv::Mat getDescriptors();

	void matchTo(cv::Mat queryDescriptors, std::vector<cv::DMatch> &matches);

private:
	char *filename;
	cv::Mat image;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	void loadImage(float f = 0.15f);
	void process();
};