#include "BPROSFeature.hpp"

BPROSFeature::BPROSFeature(int bh, int bw):
	basicFeatures(NUM_COLORS){
	blockHeight = bh;
	blockWidth = bw;

	num_Rows = ((SCREEN_HEIGHT % blockHeight == 0) ? SCREEN_HEIGHT/blockHeight : (SCREEN_HEIGHT/blockHeight + 1));
	num_Columns = ((SCREEN_WIDTH % blockWidth == 0) ? SCREEN_WIDTH/blockWidth : (SCREEN_WIDTH/blockWidth + 1));

    bprosFeatures.resize(NUM_COLORS);
    for (int i = 0; i < NUM_COLORS; i++){
        bprosFeatures[i].resize(NUM_COLORS);
    }
}

vector<vector<tuple<int, int>>>& BPROSFeature::getBasicFeatures(const IntMatrix& screen){
	for(int r = 0; r < num_Rows; r++){
		for(int c = 0; c < num_Columns; c++){
			vector<bool> hasColor(NUM_COLORS, false);
			for(int i = r * blockHeight; i < (r+1) * blockHeight; i++){
				for(int j = c * blockWidth; j < (c+1) * blockWidth; j++){
					if(i < SCREEN_HEIGHT && j < SCREEN_WIDTH){
						int pixel =  screen[i][j];
						if(!hasColor[pixel]){
							hasColor[pixel] = true;
							tuple<int, int> pos(r, c);
							basicFeatures[pixel].push_back(pos);
						}
					}
				}
			}
		}
	}
	return basicFeatures;
}

void BPROSFeature::addRelativeFeatures( vector<vector<tuple<int,int> > > &basicFeatures){	
    for(int c1 = 0;c1 < NUM_COLORS; c1++){
        for(int c2 = c1; c2 < NUM_COLORS; c2++){
            if(basicFeatures[c1].size() > 0 && basicFeatures[c2].size() > 0){
                for(vector<tuple<int,int> >::iterator it1 = basicFeatures[c1].begin(); it1 != basicFeatures[c1].end(); it1++){
                	vector<tuple<int,int> >::iterator it2;
                	if(c1 == c2)
                		it2 = it1;
                	else
                		it2 = basicFeatures[c2].begin();
                    for(; it2 != basicFeatures[c2].end(); it2++){
                        int rowDelta = get<0>(*it1) - get<0>(*it2) + num_Rows - 1;
                        int columnDelta = get<1>(*it1) - get<1>(*it2) + num_Columns - 1;

                        tuple<int,int> relativePos(rowDelta, columnDelta);
                        bool newPair = true;
                        for(vector<tuple<int,int>>::iterator it3 = bprosFeatures[c1][c2].begin(); it3 != bprosFeatures[c1][c2].end(); it3++){
                        	if( (*it3) == relativePos){
                        		newPair = false;
                        		break;
                        	}
                        }
                        if(newPair)
                        	bprosFeatures[c1][c2].push_back(relativePos);
                    }
                }
            }
        }
    }
}

const vector<vector<vector<tuple<int, int>>>>& BPROSFeature::getBprosFeatures(const IntMatrix& screen){
	featureSetsClear();
	addRelativeFeatures(getBasicFeatures(screen));
	return bprosFeatures;
}

void BPROSFeature::featureSetsClear(){
	for(int pixel = 0; pixel < NUM_COLORS; pixel ++)
		basicFeatures[pixel].clear();

	for (int c1 = 0; c1 < NUM_COLORS; c1++){
		for( int c2 = 0; c2 < NUM_COLORS; c2++){
			bprosFeatures[c1][c2].clear();
		}
    }
}
