#include "EFeatureIW1Search.hpp"
#include "SearchAgent.hpp"
#include <list>

EFeatureIW1Search::EFeatureIW1Search(RomSettings *rom_settings, Settings &settings,
			       ActionVect &actions, StellaEnvironment* _env) :
	SearchTree(rom_settings, settings, actions, _env) {
	
	m_stop_on_first_reward = settings.getBool( "iw1_stop_on_first_reward", true );

	int val = settings.getInt( "iw1_reward_horizon", -1 );

	m_reward_horizon = ( val < 0 ? std::numeric_limits<unsigned>::max() : val ); 

	/** Added by xhy*/
	m_display = settings.get_OSystem()->p_display_screen;

	/** Modified by xhy */
	std::cout<<"----Use the E-feature IW1 Search!----"<<std::endl;
	m_bprosFeature = new MhtBPROSFeature(5, 5); // divide screen into 14 x 16 tiles of size 15 x 10 pixels
	m_novelty_table = new bool [m_bprosFeature->get_basicFeatureSize() + m_bprosFeature->get_bprosFeatureSize()];
	e_features = new unsigned [m_bprosFeature->get_basicFeatureSize() + m_bprosFeature->get_bprosFeatureSize()];
}

EFeatureIW1Search::~EFeatureIW1Search() {
	delete m_bprosFeature;
	delete [] m_novelty_table;
	delete [] e_features;
}

/* *********************************************************************
   Builds a new tree
   ******************************************************************* */
void EFeatureIW1Search::build(ALEState & state) {	
	assert(p_root == NULL);
	p_root = new TreeNode(NULL, state, NULL, UNDEFINED, 0);
	update_tree();
	is_built = true;
}

void EFeatureIW1Search::print_path(TreeNode * node, int a) {
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

void EFeatureIW1Search::update_tree() {
	if(p_root->accumulated_reward < 0)
		p_root->accumulated_reward = 0;
	expand_tree(p_root);
}

void EFeatureIW1Search::update_novelty_table( const MhtBPROSFeature* m_bprosFeature, int depth){
	const vector<int>& novelty_true_pos = m_bprosFeature->novel_true_pos;
	for( unsigned k = 0; k < novelty_true_pos.size(); k++){
		int pos = novelty_true_pos[k];
		m_novelty_table[pos] = true;
		e_features[pos] = depth;
	}
}

bool EFeatureIW1Search::check_novelty_1(MhtBPROSFeature* m_bprosFeature, int depth){
	const vector<vector<tuple<int, int>>>& basicFeatures = m_bprosFeature->getBasicFeatures();
	const vector<vector<vector<int>>>& bprosFeatures = m_bprosFeature->getBprosFeatures();
	m_bprosFeature->novel_true_pos.clear();

	bool novelty = false;
	for( int c = 0; c < BPROS_NUM_COLORS; c++){
		for( unsigned k = 0; k < basicFeatures[c].size(); k++){
			int pos = (c * m_bprosFeature->n_rows() + get<0>(basicFeatures[c][k])) * m_bprosFeature->n_cols() + get<1>(basicFeatures[c][k]);
			if(!m_novelty_table[pos] || depth<e_features[pos]){
				m_bprosFeature->novel_true_pos.push_back(pos);
				novelty = true;
			}
		}
	}

	for( unsigned c1 = 0; c1 < bprosFeatures.size(); c1++){
		for( unsigned c2 = 0; c2 < bprosFeatures[c1].size(); c2++){
			for( unsigned k = 0; k < bprosFeatures[c1][c2].size(); k++){
				int pos = m_bprosFeature->get_basicFeatureSize() + (c1 * BPROS_NUM_COLORS + c2) * (m_bprosFeature->n_rows() + m_bprosFeature->n_cols()) + bprosFeatures[c1][c2][k];
				if( !m_novelty_table[pos] || depth<e_features[pos]){
					m_bprosFeature->novel_true_pos.push_back(pos);
					novelty = true;
				}
			}
		}
	}
	return novelty;
}

void EFeatureIW1Search::checkAndUpdate_novelty(TreeNode * curr_node, TreeNode * child, int a){
	const IntMatrix& subtracted_screen = m_display->subtractBg(child->state.getScreen());
	m_bprosFeature->getFeaturesFromScreen(subtracted_screen);
	if( check_novelty_1(m_bprosFeature, child->m_depth)){
		update_novelty_table(m_bprosFeature, child->m_depth);
		child->is_terminal = false;
	}
	else{
		curr_node->v_children[a] = child;
		child->is_terminal = true;
		m_pruned_nodes++;
	}
}

int EFeatureIW1Search::expand_node( TreeNode* curr_node, queue<TreeNode*>& q )
{
	int num_simulated_steps = 0;
	int num_actions = available_actions.size();
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
void EFeatureIW1Search::expand_tree(TreeNode* start_node) {
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
	if(m_display){
		const IntMatrix& subtracted_screen = m_display->subtractBg(start_node->state.getScreen());
		m_bprosFeature->getFeaturesFromScreen(subtracted_screen);
		check_novelty_1(m_bprosFeature, 1);
		update_novelty_table(m_bprosFeature, 1);
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
			else 
				steps = 0;
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

void EFeatureIW1Search::clear()
{
	SearchTree::clear();

	memset(m_novelty_table, 0, m_bprosFeature->get_bprosFeatureSize()*sizeof(bool));
	memset(e_features, 1, m_bprosFeature->get_bprosFeatureSize()*sizeof(unsigned));
}

void EFeatureIW1Search::move_to_best_sub_branch() 
{
	SearchTree::move_to_best_sub_branch();

	memset(m_novelty_table, 0, m_bprosFeature->get_bprosFeatureSize()*sizeof(bool));
	memset(e_features, 1, m_bprosFeature->get_bprosFeatureSize()*sizeof(unsigned));
}

/* *********************************************************************
   Updates the branch reward for the given node
   which equals to: node_reward + max(children.branch_return)
   ******************************************************************* */
void EFeatureIW1Search::update_branch_return(TreeNode* node) {
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

void EFeatureIW1Search::set_terminal_root(TreeNode * node) {
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

void EFeatureIW1Search::print_frame_data( int frame_number, float elapsed, Action curr_action, std::ostream& output )
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
