#include "MhtBPROSFeature.hpp"

MhtBPROSFeature::MhtBPROSFeature(int bh, int bw):
	basicFeatures(BPROS_NUM_COLORS),
	n_color(0),
	blockHeight(bh),
	blockWidth(bw){
	num_Rows = ((SCREEN_HEIGHT % blockHeight == 0) ? SCREEN_HEIGHT/blockHeight : (SCREEN_HEIGHT/blockHeight + 1));
	num_Columns = ((SCREEN_WIDTH % blockWidth == 0) ? SCREEN_WIDTH/blockWidth : (SCREEN_WIDTH/blockWidth + 1));

	basicFeatureSize = BPROS_NUM_COLORS * num_Rows * num_Columns;
	bprosFeatureSize = BPROS_NUM_COLORS * BPROS_NUM_COLORS * (num_Rows + num_Columns);

    bprosFeatures.resize(BPROS_NUM_COLORS);
    for (int i = 0; i < BPROS_NUM_COLORS; i++){
        bprosFeatures[i].resize(BPROS_NUM_COLORS);
    }

    for(int i = 0; i < NUM_COLORS; i++)
    	color_map[i] = -1;
    n_color = 0;

	std::cout<<"----Use the Manhattan Bpros feature!----"<<std::endl;
}

vector<vector<tuple<int, int>>>& MhtBPROSFeature::getBasicFeatures(const IntMatrix& screen){
	for(int pixel = 0; pixel < BPROS_NUM_COLORS; pixel ++)
		basicFeatures[pixel].clear();

	for(int r = 0; r < num_Rows; r++){
		for(int c = 0; c < num_Columns; c++){
			vector<bool> hasColor(BPROS_NUM_COLORS, false);
			for(int i = r * blockHeight; i < (r+1) * blockHeight; i++){
				for(int j = c * blockWidth; j < (c+1) * blockWidth; j++){
					if(i < SCREEN_HEIGHT && j < SCREEN_WIDTH){
						int pixel =  screen[i][j];
						if(pixel == 0)
							continue;
						if(color_map[pixel] == -1){
							if(n_color >= BPROS_NUM_COLORS){
								std::cout << "Not enough colors for Bpros features..." << " n_color:"<<n_color<<",Bpros colors:" << BPROS_NUM_COLORS << std::endl;
								exit(-1);
							}
							color_map[pixel] = n_color++;
						}
						pixel = color_map[pixel];

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

void MhtBPROSFeature::addRelativeFeatures( vector<vector<tuple<int,int>> > &basicFeatures){	
	for (int c1 = 0; c1 < BPROS_NUM_COLORS; c1++){
		for( int c2 = 0; c2 < BPROS_NUM_COLORS; c2++){
			bprosFeatures[c1][c2].clear();
		}
    }

    for(int c1 = 0;c1 < BPROS_NUM_COLORS; c1++){
        for(int c2 = c1+1; c2 < BPROS_NUM_COLORS; c2++){
            if(basicFeatures[c1].size() > 0 && basicFeatures[c2].size() > 0){
                for(vector<tuple<int,int> >::iterator it1 = basicFeatures[c1].begin(); it1 != basicFeatures[c1].end(); it1++){

                    for(vector<tuple<int,int> >::iterator it2 = basicFeatures[c2].begin(); it2 != basicFeatures[c2].end(); it2++){
						int rowDelta = get<0>(*it1) - get<0>(*it2);
                        int columnDelta = get<1>(*it1) - get<1>(*it2);
                        rowDelta = rowDelta>0? rowDelta:-rowDelta;
                        columnDelta = columnDelta>0? columnDelta:-columnDelta;

                        int relativeDis = rowDelta+columnDelta;
                        bool newPair = true;
                        for(vector<int>::iterator it3 = bprosFeatures[c1][c2].begin(); it3 != bprosFeatures[c1][c2].end(); it3++){
                        	if( (*it3) == relativeDis){
                        		newPair = false;
                        		break;
                        	}
                        }
                        if(newPair)
                        	bprosFeatures[c1][c2].push_back(relativeDis);
                    }
                }
            }
        }
    }
}

void MhtBPROSFeature::getFeaturesFromScreen(const IntMatrix& screen){
	addRelativeFeatures(getBasicFeatures(screen));
}
