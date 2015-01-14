// 
// Stores matches between image A and B.
// Can transform a position in image A to a position in image B.
//

#include <vector>
#include <opencv2/features2d/features2d.hpp>


class Transformer
{
public:
	struct Connection;

	template<typename T>
	struct Position
	{
		T x, y;
	};

	Transformer(std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> &queryKeypoints, std::vector<cv::KeyPoint> &trainKeypoints); //trainkeypoints are keypoints from image A,  queryKeypoints are from B
	~Transformer();
	
	Position<int> transformPosition(int x, int y);
private:
	int numOfConnections;
	Connection *connections;

	void createConnection(cv::KeyPoint from, cv::KeyPoint to, cv::DMatch);
};