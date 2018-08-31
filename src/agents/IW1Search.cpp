#include "IW1Search.hpp"
#include "SearchAgent.hpp"
#include <list>

IW1Search::IW1Search(RomSettings *rom_settings, Settings &settings,
			       ActionVect &actions, StellaEnvironment* _env) :
	SearchTree(rom_settings, settings, actions, _env) {
	
	m_stop_on_first_reward = settings.getBool( "iw1_stop_on_first_reward", true );

	int val = settings.getInt( "iw1_reward_horizon", -1 );

	m_novelty_boolean_representation = settings.getBool( "novelty_boolean", false );

	m_reward_horizon = ( val < 0 ? std::numeric_limits<unsigned>::max() : val ); 

	/** Added by xhy*/
	m_display = settings.get_OSystem()->p_display_screen;

	/** Added by xhy, if true, apply game screens instead of ALE_RAM to IW1 */
	m_screen_features_on = settings.getBool("screen_features_on", false);
	m_bpros_features = settings.getBool("bpros_features", false);
	m_ae_features = settings.getBool("ae_features", false);
	m_seq_ae_features = settings.getBool("seq_ae_features", false);
	

	/** Modified by xhy */
	if(m_bpros_features){
		std::cout<<"Bpros features on!"<<std::endl;
		m_bprosFeature = new BPROSFeature(5, 5); // divide screen into 14 x 16 tiles of size 15 x 10 pixels
		// m_ram_novelty_table_true = new Bit_Matrix( m_bprosFeature->get_basicFeatureSize() + m_bprosFeature->get_bprosFeatureSize(), 1);
		m_ram_novelty_table_true = new Bit_Matrix(m_bprosFeature->get_basicFeatureSize(), 1);
		//m_ram_novelty_table_false = new Bit_Matrix( m_bprosFeature->get_basicFeatureSize() + m_bprosFeature->get_bprosFeatureSize(), 1);
		m_ram_novelty_table_false = new Bit_Matrix(1,1);
	}
	else if(m_ae_features){
		std::cout << "Loading cr_predictor..." << std::endl;
		m_ae = new cs_predictor();
		std::cout << "Loaded!" << std::endl;

		if(m_novelty_boolean_representation){
			m_ram_novelty_table_true = new Bit_Matrix( HIDDEN_SIZE , 10 );
			m_ram_novelty_table_false = new Bit_Matrix( HIDDEN_SIZE , 10 );
		}
		else{
			m_ram_novelty_table = new Bit_Matrix( HIDDEN_SIZE , 1024);
		}
	}
	else if(m_seq_ae_features){
		
	}
	else{
		if(m_novelty_boolean_representation){
			if(!m_screen_features_on){
				m_ram_novelty_table_true = new Bit_Matrix( RAM_SIZE, 8 );
				m_ram_novelty_table_false = new Bit_Matrix( RAM_SIZE, 8 );
			}
			else{
				m_ram_novelty_table_true = new Bit_Matrix( SCREEN_SIZE, 8 );
				m_ram_novelty_table_false = new Bit_Matrix( SCREEN_SIZE, 8 );
			}
		}
		else{
			if(!m_screen_features_on)
				m_ram_novelty_table = new Bit_Matrix( RAM_SIZE, NUM_COLORS );
			else
				m_ram_novelty_table = new Bit_Matrix( SCREEN_SIZE, NUM_COLORS);
		}
	}
}

IW1Search::~IW1Search() {
	if(m_novelty_boolean_representation){
		delete m_ram_novelty_table_true;
		delete m_ram_novelty_table_false;
	}
	else
		delete m_ram_novelty_table;

	if(m_bprosFeature)
		delete m_bprosFeature;

	if(m_ae)
		delete m_ae;
}

/* *********************************************************************
   Builds a new tree
   ******************************************************************* */
void IW1Search::build(ALEState & state) {	
	assert(p_root == NULL);
	p_root = new TreeNode(NULL, state, NULL, UNDEFINED, 0);
	update_tree();
	is_built = true;
}

void IW1Search::print_path(TreeNode * node, int a) {
	cerr << "Path, return " << node->v_children[a]->branch_return << endl;

	while (!node->is_leaf()) {
		TreeNode * child = node->v_children[a];

		cerr << "\tAction " << a << " Reward " << child->node_reward << 
			" Return " << child->branch_return << 
			" Terminal " << child->is_terminal << endl;
    
		node = child;
		if (!node->is_leaf())
			a = node->best_branch;
	}
}

// void IW1Search::initialize_tree(TreeNode* node){
// 	do {
// 		Action act =  Action::PLAYER_A_NOOP
// 		m_generated_nodes++;

// 		TreeNode* child = new TreeNode(	curr_node,	
// 					curr_node->state,
// 					this,
// 					act,
// 					sim_steps_per_node); 
		
// 		if ( check_novelty_1( child->state.getRAM() ) ) {
// 			update_novelty_table( child->state.getRAM() );
			
// 		}
// 		else{
// 			curr_node->v_children[a] = child;
// 			child->is_terminal = true;
// 			m_pruned_nodes++;
// 			break;
// 		}

// 		num_simulated_steps += child->num_simulated_steps;					
// 		node->v_children[a] = child;
		
// 		node = child;
// 	}while( node->depth() < m_max_depth)
// }

void IW1Search::update_tree() {
	if(m_seq_ae_features)
		p_root->set_last5_screens(m_env->last5_screens, m_env->update_pos);

	if(p_root->accumulated_reward < 0)
		p_root->accumulated_reward = 0;
	expand_tree(p_root);
	// for(unsigned byte = 0; byte < RAM_SIZE; byte++){
	//     std::cout << "Byte: " << byte << std::endl;
	//     int count = 0;
	//     for( unsigned i = 0; i < 255; i++){
	// 	if ( m_ram_novelty_table->iset( byte,i ) ){
	// 	    count++;			
	// 	    std::cout << "\t element: "<< i << std::endl;
	// 	}

	//     }
	//     std::cout << "\t num_elements " << count << std::endl;
	// }
	// std::exit(0);
}

void IW1Search::update_novelty_table( const ALERAM& machine_state )
{
	for ( size_t i = 0; i < machine_state.size(); i++ )
		if( m_novelty_boolean_representation ){
			unsigned char mask = 1;
			byte_t byte =  machine_state.get(i);
			for(int j = 0; j < 8; j++) {
				bool bit_is_set = (byte & (mask << j)) != 0;
				if( bit_is_set )
					m_ram_novelty_table_true->set( i, j );
				else
					m_ram_novelty_table_false->set( i, j );
			}
		}
		else
			m_ram_novelty_table->set( i, machine_state.get(i) );
}

/** Added by xhy*/
void IW1Search::update_novelty_table( const ALEScreen& curr_alescreen){
	for( int i = 0; i < SCREEN_HEIGHT; i++){
		for( int j = 0; j < SCREEN_WIDTH; j++){
			int pixel = curr_alescreen.get(i,j);
			if(m_novelty_boolean_representation){
				unsigned char mask = 1;
				for( int k = 0; k < 8; k++){
					bool bit_is_set = (pixel & (mask << k)) != 0;
					if(bit_is_set)
						m_ram_novelty_table_true->set( i * SCREEN_WIDTH + j, k);
					else
						m_ram_novelty_table_false->set( i * SCREEN_WIDTH + j, k);
				}
			}
			else
				m_ram_novelty_table->set(i * SCREEN_WIDTH + j, pixel );
		}
	}
}

/** Added by xhy, to update novelty table with subtracted game screen*/
void IW1Search::update_novelty_table( const IntMatrix &screen){
	for( int i = 0; i < SCREEN_HEIGHT; i++){
		for( int j = 0; j < SCREEN_WIDTH; j++){
			int pixel = screen[i][j];
			//if(screen[i][j] != 0){//only pixles with non-zero value(not background) should be considered
				if(m_novelty_boolean_representation){
					unsigned char mask = 1;
					for( int k = 0; k < 8; k++){
						bool bit_is_set = (pixel & (mask << k)) != 0;
						if(bit_is_set)
							m_ram_novelty_table_true->set( i * SCREEN_WIDTH + j, k);
						else
							m_ram_novelty_table_false->set( i * SCREEN_WIDTH + j, k);
					}
				}
				else
					m_ram_novelty_table->set(i * SCREEN_WIDTH + j, pixel );
			//}
		}
	}
}

void IW1Search::update_novelty_table( const BPROSFeature* m_bprosFeature){
	const vector<int>& novelty_true_pos = m_bprosFeature->novel_true_pos;
	//const vector<int>& novelty_false_pos = m_bprosFeature->novel_false_pos;

	for( unsigned k = 0; k < novelty_true_pos.size(); k++){
		int pos = novelty_true_pos[k];
		m_ram_novelty_table_true->set( pos , 1 );
	}
	// for( unsigned k = 0; k < novelty_false_pos.size(); k++){
	// 	int pos = novelty_false_pos[k];
	// 	m_ram_novelty_table_false->set( pos , 1 );
	// }
}

void IW1Search::update_novelty_table(const int* hidden_state){
	int novelty_boolean_size = 12;
	if(m_ae_features){
		novelty_boolean_size = 10;
	}
	else if(m_seq_ae_features){
		// novelty_boolean_size = 8;
	}

	for(int i = 0; i < HIDDEN_SIZE; i++){
		int state = hidden_state[i];
		if(m_novelty_boolean_representation){
			unsigned char mask = 1;
			for( int k = 0; k < novelty_boolean_size; k++){
				bool bit_is_set = (state & (mask << k)) != 0;
				if(bit_is_set)
					m_ram_novelty_table_true->set(i, k);
				else
					m_ram_novelty_table_false->set(i, k);
			}
		}
		else
			m_ram_novelty_table->set(i, state);
	}
}

bool IW1Search::check_novelty_1( const ALERAM& machine_state )
{
	for ( size_t i = 0; i < machine_state.size(); i++ )	
		if( m_novelty_boolean_representation ){
			unsigned char mask = 1;
			byte_t byte =  machine_state.get(i);
			for(int j = 0; j < 8; j++) {
				bool bit_is_set = (byte & (mask << j)) != 0;
				if( bit_is_set ){
					if( ! m_ram_novelty_table_true->iset( i, j ) )
						return true;
				}
				else{
					if( ! m_ram_novelty_table_false->iset( i, j ) )
						return true;
				}
			}
		}
		else{
			if ( ! m_ram_novelty_table->iset( i, machine_state.get(i) ) )
				return true;
		}
	return false;
}

/** Added by xhy*/
bool IW1Search::check_novelty_1( const ALEScreen& curr_alescreen)
{
	for( int i = 0; i < SCREEN_HEIGHT; i++){
		for( int j = 0; j < SCREEN_WIDTH; j++){
			int pixel = curr_alescreen.get(i, j);
			if(m_novelty_boolean_representation){
				unsigned char mask = 1;
				for( int k = 0; k < 8; k++) {
                    bool bit_is_set = (pixel & (mask << k)) != 0;
                    if (bit_is_set) {
                        if (!m_ram_novelty_table_true->iset(i * SCREEN_WIDTH + j, k))
                            return true;
                    }
					else{
                        if (!m_ram_novelty_table_false->iset(i * SCREEN_WIDTH + j, k))
                            return true;
                        }
				}
			}
			else{
				if ( !m_ram_novelty_table->iset( i * SCREEN_WIDTH + j, pixel ) )
					return true;
			}
		}
	}
	return false;
}

/** Added by xhy*/
bool IW1Search::check_novelty_1( const IntMatrix &screen)
{
	for( int i = 0; i < SCREEN_HEIGHT; i++){
		for( int j = 0; j < SCREEN_WIDTH; j++){
			int pixel = screen[i][j];
			//if(screen[i][j] != 0){//only pixles with non-zero value should be considered
				if(m_novelty_boolean_representation){
					unsigned char mask = 1;
					for( int k = 0; k < 8; k++) {
                        bool bit_is_set = (pixel & (mask << k)) != 0;
                        if (bit_is_set) {
                            if (!m_ram_novelty_table_true->iset(i * SCREEN_WIDTH + j, k))
                                return true;
                        }
						else {
                                if (!m_ram_novelty_table_false->iset(i * SCREEN_WIDTH + j, k))
                                    return true;
                        }
					}
				}
				else{
					if ( !m_ram_novelty_table->iset( i * SCREEN_WIDTH + j, pixel ) )
						return true;
				}
			//}
		}
	}
	return false;
}

// bool IW1Search::check_novelty_1(BPROSFeature* m_bprosFeature){
// 	const vector<vector<tuple<int, int>>>& basicFeatures = m_bprosFeature->getBasicFeatures();
// 	const vector<vector<vector<tuple<int, int>>>>& bprosFeatures = m_bprosFeature->getBprosFeatures();
// 	m_bprosFeature->novel_true_pos.clear();
// 	m_bprosFeature->novel_false_pos.clear();

// 	bool novelty = false;
// 	for( int c = 0; c < BPROS_NUM_COLORS; c++){
// 		for( unsigned k = 0; k < basicFeatures[c].size(); k++){
// 			int pos = (c * m_bprosFeature->n_rows() + get<0>(basicFeatures[c][k])) * m_bprosFeature->n_cols() + get<1>(basicFeatures[c][k]);
// 			if(!m_ram_novelty_table_true->iset( pos, 1)){
// 				m_bprosFeature->novel_true_pos.push_back(pos);
// 				novelty = true;
// 			}
// 			else if(!m_ram_novelty_table_false->iset( pos, 1)){
// 				m_bprosFeature->novel_false_pos.push_back(pos);
// 				novelty = true;
// 			}
// 		}
// 	}

// 	for( unsigned c1 = 0; c1 < bprosFeatures.size(); c1++){
// 		for( unsigned c2 = 0; c2 < bprosFeatures[c1].size(); c2++){
// 			for( unsigned k = 0; k < bprosFeatures[c1][c2].size(); k++){
// 				int pos = m_bprosFeature->get_basicFeatureSize() + ((c1 * BPROS_NUM_COLORS + c2) * m_bprosFeature->n_rows() + get<0>(bprosFeatures[c1][c2][k])) * m_bprosFeature->n_cols() + get<1>(bprosFeatures[c1][c2][k]);
// 				if( !m_ram_novelty_table_true->iset(pos , 1) ){
// 					m_bprosFeature->novel_true_pos.push_back(pos);
// 					novelty = true;
// 				}
// 				else if(!m_ram_novelty_table_false->iset( pos, 1)){
// 					m_bprosFeature->novel_false_pos.push_back(pos);
// 					novelty = true;
// 				}
// 			}
// 		}
// 	}
// 	return novelty;
// }

bool IW1Search::check_novelty_1(BPROSFeature* m_bprosFeature){
	const vector<vector<tuple<int, int>>>& basicFeatures = m_bprosFeature->getBasicFeatures();
	// const vector<vector<vector<int>>>& bprosFeatures = m_bprosFeature->getBprosFeatures();
	m_bprosFeature->novel_true_pos.clear();
	//m_bprosFeature->novel_false_pos.clear();

	bool novelty = false;
	for( int c = 0; c < BPROS_NUM_COLORS; c++){
		for( unsigned k = 0; k < basicFeatures[c].size(); k++){
			int pos = (c * m_bprosFeature->n_rows() + get<0>(basicFeatures[c][k])) * m_bprosFeature->n_cols() + get<1>(basicFeatures[c][k]);
			if(!m_ram_novelty_table_true->iset( pos, 1)){
				m_bprosFeature->novel_true_pos.push_back(pos);
				novelty = true;
			}
			// else if(!m_ram_novelty_table_false->iset( pos, 1)){
			// 	m_bprosFeature->novel_false_pos.push_back(pos);
			// 	novelty = true;
			// }
		}
	}

	// for( unsigned c1 = 0; c1 < bprosFeatures.size(); c1++){
	// 	for( unsigned c2 = 0; c2 < bprosFeatures[c1].size(); c2++){
	// 		for( unsigned k = 0; k < bprosFeatures[c1][c2].size(); k++){
	// 			int pos = m_bprosFeature->get_basicFeatureSize() + (c1 * BPROS_NUM_COLORS + c2) * (m_bprosFeature->n_rows() + m_bprosFeature->n_cols()) + bprosFeatures[c1][c2][k];
	// 			// int pos = (c1 * BPROS_NUM_COLORS + c2) * (m_bprosFeature->n_rows() + m_bprosFeature->n_cols()) + bprosFeatures[c1][c2][k];
	// 			if( !m_ram_novelty_table_true->iset(pos , 1) ){
	// 				m_bprosFeature->novel_true_pos.push_back(pos);
	// 				novelty = true;
	// 			}
	// 			// else if(!m_ram_novelty_table_false->iset( pos, 1)){
	// 			// 	m_bprosFeature->novel_false_pos.push_back(pos);
	// 			// 	novelty = true;
	// 			// }
	// 		}
	// 	}
	// }
	return novelty;
}

bool IW1Search::check_novelty_1(const int* hidden_state){
	int novelty_boolean_size = 8;
	if(m_ae_features){
		novelty_boolean_size = 10;
	}
	else if(m_seq_ae_features){
		// hidden_state_size = HIDDEN1_SIZE;
		// novelty_boolean_size = 8;
	}

	for(int i = 0; i < HIDDEN_SIZE; i++){
		int state = hidden_state[i];
		if(m_novelty_boolean_representation){
			unsigned char mask = 1;
			for( int k = 0; k < novelty_boolean_size; k++) {
	            bool bit_is_set = (state & (mask << k)) != 0;
	            if (bit_is_set) {
	                if (!m_ram_novelty_table_true->iset(i, k))
	                    return true;
	            }
				else{
	                if (!m_ram_novelty_table_false->iset(i, k))
	                    return true;
	            }
			}
		}
		else{
			if ( !m_ram_novelty_table->iset(i, state ) )
				return true;
		}
	}
	return false;
}

void IW1Search::checkAndUpdate_novelty(TreeNode * curr_node, TreeNode * child, int a){
	if(m_bpros_features){
		const IntMatrix& subtracted_screen = m_display->subtractBg(child->state.getScreen());
		m_bprosFeature->getFeaturesFromScreen(subtracted_screen);
		if( check_novelty_1(m_bprosFeature)){
			update_novelty_table(m_bprosFeature);
			child->is_terminal = false;
		}
		else{
			curr_node->v_children[a] = child;
			child->is_terminal = true;
			m_pruned_nodes++;
			//continue;
		}
	}
	else if(m_ae_features){
		if(m_display){
			m_ae->predict(m_display->subtractBg(child->state.getScreen()));
			const int* hidden_state = m_ae->get_hidden1();
			if( check_novelty_1(hidden_state)){
				update_novelty_table(hidden_state);
				child->is_terminal = false;
			}
			else{
				curr_node->v_children[a] = child;
				child->is_terminal = true;
				m_pruned_nodes++;
				//continue;	
			}
		}
	}
	else if(m_seq_ae_features){
		
	}
	else{
		if(!m_screen_features_on){
			if ( check_novelty_1( child->state.getRAM() ) ) {
				update_novelty_table( child->state.getRAM() );
				child->is_terminal = false;			
			}
			else{
				curr_node->v_children[a] = child;
				child->is_terminal = true;
				m_pruned_nodes++;
				//continue;				
			}
		}
		else{
			if(m_display){
				const IntMatrix& subtracted_screen = m_display->subtractBg(child->state.getScreen());
				// const IntMatrix& screen_diff = m_display->getDiff(curr_node->state.getScreen(), child->state.getScreen());
				if( check_novelty_1(subtracted_screen)){
					update_novelty_table(subtracted_screen);
					child->is_terminal = false;
				}
				else{
					curr_node->v_children[a] = child;
					child->is_terminal = true;
					m_pruned_nodes++;
					//continue;				
				}
			}
		}
	}
}

int IW1Search::expand_node( TreeNode* curr_node, queue<TreeNode*>& q )
{
	int num_simulated_steps =0;
	int num_actions = available_actions.size();
	//std::cout << "num_actions" << num_actions << std::endl;
	bool leaf_node = (curr_node->v_children.empty());
	static unsigned max_nodes_per_frame = max_sim_steps_per_frame / sim_steps_per_node;
	m_expanded_nodes++;
	// Expand all of its children (simulates the result)
	if(leaf_node){
		curr_node->v_children.resize( num_actions );
		curr_node->available_actions = available_actions;
		if(m_randomize_successor)
		    std::random_shuffle(curr_node->available_actions.begin(), curr_node->available_actions.end());
	
	}

	for (int a = 0; a < num_actions; a++) {
		Action act = curr_node->available_actions[a];
		
		TreeNode * child;
	
		// If re-expanding an internal node, don't creates new nodes
		if (leaf_node) {
			m_generated_nodes++;
			child = new TreeNode(	curr_node,	
						curr_node->state,
						this,
						act,
						sim_steps_per_node);

			// if(m_seq_ae_features){
			// 	child->set_last5_screens(curr_node->last5_screens, curr_node->update_pos);
			// 	child->update_last5_screens();
			// }

			checkAndUpdate_novelty(curr_node, child, a);
			if (child->depth() > m_max_depth ) 
				m_max_depth = child->depth();
			num_simulated_steps += child->num_simulated_steps;
					
			curr_node->v_children[a] = child;
		}
		else {
			child = curr_node->v_children[a];
			// This recreates the novelty table (which gets resetted every time
			// we change the root of the search tree)
			if ( m_novelty_pruning )
			{
			    if( child->is_terminal )
			    	checkAndUpdate_novelty(curr_node, child, a);
			}
			child->updateTreeNode();
			if (child->depth() > m_max_depth ) m_max_depth = child->depth();

			// DONT COUNT REUSED NODES
			//if ( !child->is_terminal )
			//	num_simulated_steps += child->num_simulated_steps;

		}
		// Don't expand duplicate nodes, or terminal nodes
		if (!child->is_terminal) {
			if (! (ignore_duplicates && test_duplicate(child)) )
			    if( child->num_nodes_reusable < max_nodes_per_frame )
					q.push(child);
		}
	}
	return num_simulated_steps;
}

/* *********************************************************************
   Expands the tree from the given node until i_max_sim_steps_per_frame
   is reached
	
   ******************************************************************* */
void IW1Search::expand_tree(TreeNode* start_node) {
	if(!start_node->v_children.empty()){
	    start_node->updateTreeNode();
	    for (unsigned a = 0; a < available_actions.size(); a++) {
			TreeNode* child = start_node->v_children[a];
			if( !child->is_terminal ){
			        child->num_nodes_reusable = child->num_nodes();
			}
	    }
	}

	queue<TreeNode*> q;
	std::list< TreeNode* > pivots;
	
	//q.push(start_node);
	assert(start_node!=NULL);
	pivots.push_back( start_node );

	/** modified by xhy*/
	if(m_bpros_features){	
		if(m_display){
			const IntMatrix& subtracted_screen = m_display->subtractBg(start_node->state.getScreen());
			m_bprosFeature->getFeaturesFromScreen(subtracted_screen);
			check_novelty_1(m_bprosFeature);
			update_novelty_table(m_bprosFeature);
		}
	}
	else if(m_ae_features){
		m_ae->predict(m_display->subtractBg(start_node->state.getScreen()));
		const int* hidden_state = m_ae->get_hidden1();
		update_novelty_table(hidden_state);
	}
	else if(m_seq_ae_features){
		
	}
	else{
		if(!m_screen_features_on){	
			update_novelty_table( start_node->state.getRAM() );
		}
		else{
			if(m_display){
				const IntMatrix& subtracted_screen = m_display->subtractBg(start_node->state.getScreen());
				//const IntMatrix& screen_diff = m_display->getDiff(start_node->state.getScreen(), start_node->state.getScreen());
				update_novelty_table(subtracted_screen);
			}
		}
	}

	int num_simulated_steps = 0;

	m_expanded_nodes = 0;
	m_generated_nodes = 0;

	m_pruned_nodes = 0;

	do {
		
		std::cout << "# Pivots size: " << pivots.size() << std::endl;
		std::cout << "First pivot reward: " << pivots.front()->node_reward << std::endl;
		pivots.front()->m_depth = 0;
		int steps = expand_node( pivots.front(), q );
		num_simulated_steps += steps;

		if (num_simulated_steps >= max_sim_steps_per_frame) {
			break;
		}

		pivots.pop_front();

		while(!q.empty()) {
			// Pop a node to expand
			TreeNode* curr_node = q.front();
			q.pop();
	
			if ( curr_node->depth() > m_reward_horizon - 1 ) continue;
			if ( m_stop_on_first_reward && curr_node->node_reward != 0 && curr_node->accumulated_reward > 0) 
			{
				pivots.push_back( curr_node );
				continue;
			}

			if(curr_node->accumulated_reward >= 0)
				steps = expand_node( curr_node, q );
			num_simulated_steps += steps;
			// Stop once we have simulated a maximum number of steps
			if (num_simulated_steps >= max_sim_steps_per_frame) {
				break;
			}
		
		}
		std::cout << "\tExpanded so far: " << m_expanded_nodes << std::endl;	
		std::cout << "\tPruned so far: " << m_pruned_nodes << std::endl;	
		std::cout << "\tGenerated so far: " << m_generated_nodes << std::endl;	

		if (q.empty()) std::cout << "Search Space Exhausted!" << std::endl;
		// Stop once we have simulated a maximum number of steps
		if (num_simulated_steps >= max_sim_steps_per_frame) {
			break;
		}

	} while ( !pivots.empty() );
	if( num_simulated_steps < max_sim_steps_per_frame && pivots.empty() && q.empty()){
		std::cout << "No nodes to expand..." << std::endl;
		std::cout << "No nodes to expand..." << std::endl;
		std::cout << "No nodes to expand..." << std::endl;
	}
	
	update_branch_return(start_node);
}

void IW1Search::clear()
{
	SearchTree::clear();

	if(m_novelty_boolean_representation){
		m_ram_novelty_table_true->clear();
		m_ram_novelty_table_false->clear();
	}
	else
		m_ram_novelty_table->clear();
}

void IW1Search::move_to_best_sub_branch() 
{
	SearchTree::move_to_best_sub_branch();
	if(m_novelty_boolean_representation){
		m_ram_novelty_table_true->clear();
		m_ram_novelty_table_false->clear();
	}
	else
		m_ram_novelty_table->clear();
}

/* *********************************************************************
   Updates the branch reward for the given node
   which equals to: node_reward + max(children.branch_return)
   ******************************************************************* */
void IW1Search::update_branch_return(TreeNode* node) {
	// Base case (leaf node): the return is the immediate reward
	if (node->v_children.empty()) {
		node->branch_return = node->node_reward;
		node->best_branch = -1;
		node->branch_depth = node->m_depth;
		return;
	}

	// First, we have to make sure that all the children are updated
	for (unsigned int c = 0; c < node->v_children.size(); c++) {
		TreeNode* curr_child = node->v_children[c];
			
		if (ignore_duplicates && curr_child->is_duplicate()) continue;
    
		update_branch_return(curr_child);
	}
	
	// Now that all the children are updated, we can update the branch-reward
	float best_return = -1;
	int best_branch = -1;
	return_t avg = 0;

	// Terminal nodes encur no reward beyond immediate
	if (node->is_terminal) {
		node->branch_depth = node->m_depth;
		best_return = node->node_reward;
		best_branch = 0;		
	} else {
		for (size_t a = 0; a < node->v_children.size(); a++) {	
			return_t child_return = node->v_children[a]->branch_return;
			if (best_branch == -1 || child_return > best_return) {
				best_return = child_return;
				best_branch = a;
				node->branch_depth = node->v_children[a]->branch_depth;
				avg+=child_return;
			}
			else if(child_return == best_return){
				if(child_return==0 && node->v_children[a]->branch_depth > node->branch_depth ){
					best_branch = a;
					node->branch_depth = node->v_children[a]->branch_depth;
					avg+=child_return;
				}
				else if(child_return!=0 && node->v_children[a]->branch_depth < node->branch_depth){
					best_branch = a;
					node->branch_depth = node->v_children[a]->branch_depth;
					avg+=child_return;
				}
			}
			
			if( node->v_children.size() ) avg/=node->v_children.size();
		}
	}

	node->branch_return = node->node_reward + best_return * discount_factor; 
	//node->branch_return = node->node_reward + avg * discount_factor; 
	
	node->best_branch = best_branch;
}

void IW1Search::set_terminal_root(TreeNode * node) {
	node->branch_return = node->node_reward; 

	if (node->v_children.empty()) {
		// Expand one child; add it to the node's children
		TreeNode* new_child = new TreeNode(	node, node->state, 
							this, PLAYER_A_NOOP, sim_steps_per_node);

		node->v_children.push_back(new_child);
    	}
  
    	// Now we have at least one child, set the 'best branch' to the first action
    	node->best_branch = 0; 
}

void IW1Search::print_frame_data( int frame_number, float elapsed, Action curr_action, std::ostream& output )
{
	output << "frame=" << frame_number;
	output << ",expanded=" << expanded_nodes();
	output << ",generated=" << generated_nodes();
	output << ",pruned=" << pruned();
	output << ",depth_tree=" << max_depth();
	output << ",tree_size=" <<  num_nodes(); 
	output << ",best_action=" << action_to_string( curr_action );
	output << ",branch_reward=" << get_root_value();
	output << ",elapsed=" << elapsed;
	m_rom_settings->print( output );
	output << std::endl;
}
