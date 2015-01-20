/*
Stores matches between image A and B.
Can transform a position in image A to a position in image B.
*/

#include <vector>
#include <opencv2/features2d/features2d.hpp>


class Transformer
{
public:
	// Stores Translation, Scale & Rotation for 2 matching keypoints. + quality of match
	struct Connection;

	// Position stores x and y for a position within an image
	template<typename T>
	class Position
	{
	public:
		T x, y;
		void rotate(float angle)
		{
			float newX, newY;
			newX = cos(angle)*this->x + sin(angle)*this->y;
			newY = cos(angle)*this->y - sin(angle)*this->x;
			x = newX;
			y = newY;
		}

		void scale(float factor)
		{
			x *= factor;
			y *= factor;
		}

		void add(Position<T> t)
		{
			x += t.x;
			y += t.y;
		}
		Position() 
			: x(), y()
		{}
	};
	template<>
	class Position < int >
	{
	public:
		int x, y;
		Position(Position<float> p)
		{
			x = (int)p.x;
			y = (int)p.y;
		}
	};

	Transformer(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &queryKeypoints, std::vector<cv::KeyPoint> &trainKeypoints, float imageDiagonal);
	~Transformer();
	
	Position<int> transformPosition(int x, int y);

private:
	float imageDiagonal;
	int numOfConnections;
	Connection *connections;

	void createConnection(cv::KeyPoint from, cv::KeyPoint to, cv::DMatch match, int i);
};