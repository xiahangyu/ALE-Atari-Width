#ifndef __RO_IW_SEARCH_HPP__
#define __RO_IW_SEARCH_HPP__

#include "SearchTree.hpp"
#include "bit_matrix.hxx"
#include "../environment/ale_ram.hpp"
#include "../common/Constants.h"
#include "BPROSFeature.hpp"

#include <queue>

class RolloutIW1Search : public SearchTree {
    public:
	RolloutIW1Search(RomSettings *, Settings &settings, ActionVect &actions, StellaEnvironment* _env);

	virtual ~RolloutIW1Search();

	virtual void build(ALEState & state);
		
	virtual void update_tree();
	virtual int  expand_node( TreeNode* n, queue<TreeNode*>& q ); 

	int expanded() const { return m_expanded_nodes; }
	int generated() const { return m_generated_nodes; }
	int pruned() const { return m_pruned_nodes; }
	virtual unsigned max_depth(){ 	
		for (size_t c = 0; c < p_root->v_children.size(); c++)
			if(m_max_depth <  p_root->v_children[c]->branch_depth)
				m_max_depth =   p_root->v_children[c]->branch_depth;

			return m_max_depth;
	}
	virtual	void print_frame_data( int frame_number, float elapsed, Action curr_action, std::ostream& output );
protected:	

	void print_path(TreeNode *start, int a);

	virtual void expand_tree(TreeNode* start);

	void update_branch_return(TreeNode* node);
    void set_terminal_root(TreeNode* node); 

	void 	update_novelty_table( const BPROSFeature* m_bprosFeature, int depth);
	bool 	check_novelty_1( BPROSFeature* m_bprosFeature, int depth);
	void	update_novelty_table( const ALERAM &machine_state);
	bool	check_novelty_1( const ALERAM& machine_state);

	void	checkAndUpdate_novelty(TreeNode *curr_node, TreeNode *child, int a);

	virtual void	clear();
	virtual void	move_to_best_sub_branch();
	
	ALERAM 			m_ram;
	unsigned		m_pruned_nodes;
	bool			m_stop_on_first_reward;
	unsigned		m_reward_horizon;	
	bool            m_novelty_boolean_representation;

	// added by xhy

	/** Added by xhy*/
	DisplayScreen* m_display;

	BPROSFeature* m_bprosFeature;
	bool *m_novelty_table;
	unsigned *e_features;
};



#endif // __RO_IW_SEARCH_HPP__
