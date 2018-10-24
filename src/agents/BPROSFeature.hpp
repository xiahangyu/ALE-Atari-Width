#ifndef __BPROS_Feature_HPP__
#define __BPROS_Feature_HPP__

#include "../common/Constants.h"
#include <tuple>
using namespace std;

class BPROSFeature { 
public:
	BPROSFeature(int bh, int bw);
	void getFeaturesFromScreen(const IntMatrix& screen);
	const vector<vector<tuple<int, int>>>& getBasicFeatures(){return basicFeatures;}
	const vector<vector<vector<tuple<int, int>>>>& getBprosFeatures(){return bprosFeatures;}

	int n_rows(){return num_Rows;}
	int n_cols(){return num_Columns;}
	int get_blockHeight(){return blockHeight;}
	int get_blockWidth(){return blockWidth;}
	int get_basicFeatureSize(){return basicFeatureSize;}
	int get_bprosFeatureSize(){return bprosFeatureSize;}

private:
	vector<vector<tuple<int, int>>>& getBasicFeatures(const IntMatrix& screen);
	void addRelativeFeatures(vector<vector<tuple<int,int> > > &basicFeatures);


public:
	vector<int> novel_true_pos;	//Modified in IW1Search::check_novelty_1(BPROSFeature* m_bprosFeature)
	vector<int> novel_false_pos;
	int color_map[NUM_COLORS];
    int n_color;

private:
	int blockHeight;
	int blockWidth;
	int num_Rows;
	int num_Columns;

	int basicFeatureSize;
	int bprosFeatureSize;

	vector<vector<tuple<int, int>>> basicFeatures;	//(color, blockPos) pairs
	vector<vector<vector<tuple<int, int>>>> bprosFeatures;	//(color1, color2, relativeBlockPos) pairs
};

#endif //__BPROS_Feature_HPP__