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

Transformer::Transformer(vector<DMatch> &matches, vector<KeyPoint> &queryKP, vector<KeyPoint> &trainKP, float queryImageDiagonal)
{
	imageDiagonal = queryImageDiagonal;
	
	// init connections
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
	float *weights = new float[numOfConnections]; 
	float weightSum = 0;
	
	Position<float> *positions = new Position<float>[numOfConnections];
	
	for (int i = 0; i < numOfConnections; ++i)
	{
		// calculate relative position
		positions[i].x = x - connections[i].pA.x;
		positions[i].y = y - connections[i].pA.y;

		// 1. we want to weight keypoints less, which are farer from the click
		// 2. we want to weight keypoints with a lower quality less
		weights[i] = pow((imageDiagonal - sqrt(pow(positions[i].x, 2) + pow(positions[i].y, 2))),4) / connections[i].distance;
		
		// transform point according to the connection
		positions[i].rotate(connections[i].rotation);
		positions[i].scale(connections[i].scale);
		positions[i].add(connections[i].pB);
		
		// sum up all weights
		weightSum += weights[i];
	}
	float normalizationFactor = 1 / weightSum;
	
	// build weighted average from all transformations
	Position<float> average;
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

