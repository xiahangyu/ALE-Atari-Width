#ifndef __BPROS_Feature_HPP__
#define __BPROS_Feature_HPP__

#include "../common/Constants.h"
#include <tuple>
using namespace std;

class BPROSFeature {
public:
	BPROSFeature(int bh, int bw);
	const vector<vector<vector<tuple<int, int>>>>& getBprosFeatures(const IntMatrix& screen);
	const vector<vector<vector<tuple<int, int>>>>& getBprosFeatures(){return bprosFeatures;}

	int n_rows(){return num_Rows;}
	int n_cols(){return num_Columns;}
	int get_blockHeight(){return blockHeight;}
	int get_blockWidth(){return blockWidth;}

private:
	vector<vector<tuple<int, int>>>& getBasicFeatures(const IntMatrix& screen);
	void addRelativeFeatures(vector<vector<tuple<int,int> > > &basicFeatures);
	void featureSetsClear();
private:
	int blockHeight;
	int blockWidth;
	int num_Rows;
	int num_Columns;

	vector<vector<tuple<int, int>>> basicFeatures;
	vector<vector<vector<tuple<int, int>>>> bprosFeatures;
};

#endif //__BPROS_Feature_HPP__