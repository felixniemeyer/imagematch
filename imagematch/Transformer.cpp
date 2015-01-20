#include <math.h>

#include "Transformer.h"

using namespace std;
using namespace cv;

struct Transformer::Connection
{
	Position<float> pA, pB;
	float distance;
	float rotation;
	float scale;
};

Transformer::Transformer(vector<DMatch> &matches, vector<KeyPoint> &queryKP, vector<KeyPoint> &trainKP, float diagonal)
{
	imageDiagonal = diagonal;

	numOfConnections = (int)matches.size();
	connections = new Connection[numOfConnections];
	int connectionsIndex = 0;

	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); ++it)
	{
		createConnection(queryKP[it->queryIdx], trainKP[it->trainIdx], *it, connectionsIndex);
		connectionsIndex++;
	}
}

Transformer::~Transformer()
{
	delete[] connections;
}

Transformer::Position<int> 
Transformer::transformPosition(int x, int y)
{
	float *weights = new float[numOfConnections]; //we want to weight keypoints less, which are farer from the click
	Position<float> *positions = new Position<float>[numOfConnections];
	float weightSum = 0;
	
	for (int i = 0; i < numOfConnections; ++i)
	{
		//calc weight
		positions[i].x = x - connections[i].pA.x;
		positions[i].y = y - connections[i].pA.y;
		weights[i] = pow((imageDiagonal - sqrt(pow(positions[i].x, 2) + pow(positions[i].y, 2))),4) / connections[i].distance; //weight depends on locality of keypoint and quality of match
		
		weightSum += weights[i];
		//calc position
	//	positions[i].rotate(connections[i].rotation);
	//	positions[i].scale(connections[i].scale);
		positions[i].add(connections[i].pB);
	}

	Position<float> average;
	float normalizationFactor = 1 / weightSum;
	for (int i = 0; i < numOfConnections; ++i)
	{
		positions[i].scale(weights[i] * normalizationFactor);
		average.add(positions[i]);
	}

	delete[] weights;

	return average;
}

void 
Transformer::createConnection(cv::KeyPoint from, cv::KeyPoint to, cv::DMatch match, int i)
{
	connections[i].rotation = to.angle - from.angle;
	connections[i].scale = to.size / from.size;
	connections[i].distance = match.distance;
	connections[i].pA.x = from.pt.x;
	connections[i].pA.y = from.pt.y;
	connections[i].pB.x = to.pt.x;
	connections[i].pB.y = to.pt.y;
}

