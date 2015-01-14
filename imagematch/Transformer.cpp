#include "Transformer.h"

using namespace std;
using namespace cv;

struct Transformer::Connection
{
	Position<float> pA, pB;

	float rotation;
	float scale;
};

Transformer::Transformer(vector<DMatch> matches, vector<KeyPoint> &queryKP, vector<KeyPoint> &trainKP)
{
	connections = new Connection[matches.size()];
	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); ++it)
		createConnection(queryKP[it->queryIdx], trainKP[it->trainIdx], *it);
}

Transformer::~Transformer()
{
	delete[] connections;
}

Transformer::Position<int> 
Transformer::transformPosition(int x, int y)
{
	//todo: evaluate connections and calculate new position 
	return Position<int>();
}

void 
Transformer::createConnection(cv::KeyPoint from, cv::KeyPoint to, cv::DMatch)
{
	//todo: create connection
}