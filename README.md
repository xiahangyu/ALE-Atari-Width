Arcade-Learning-Environment
===========================

The Arcade Learning Environment (ALE) -- a platform for AI research.

See http://www.arcadelearningenvironment.org for details.

Classical Planning algorithms
=============================

The Classical Planning algorithms are located in: 

  IW(1)
  
      * src/agents/IW1Search.hpp
      * src/agents/IW1Search.cpp
      
  Rollout_IW(1)
  
      * src/agents/RolloutIW1Search.hpp
      * src/agents/RolloutIW1Search.cpp

To compile, rename the makefile either for linux or mac

Screen backgrounds
===========
All screen backgrounds are saved in the directory backgrounds/.
To get backgrounds for other games, set the background variable to true in the src/controllers/internal_controller.cpp line 58 and change the background file path in the line 107 and 108. For exmaple, the background path is "./backgrounds/ms_pacman/background.png" now, if we want to get the backgroudn of game "up_and_down" simply create a directory named up_and_downs from backgrounds and change "./backgrounds/ms_pacman/background.png" to "./backgrounds/up_and_downs/background.png". The same change should be applied to line 108.

RUNNING ALE with IW(1) and RolloutIW(1)
===========
1. Loading game background. This is done in the src/controllers/internal_controller.cpp at line 60 with background set to false. Parameter for the loadBgFromMatrix should be set to the format of "./backgrounds/game_name/background.matrix".

2. Set screen feature types(Game RAM, Background subtracted game screen, Bpros game screen). Screen feature types can be set in the src\common\Default.cpp with configurations set to:

  Game RAM: 
  
    * settings.setBool("screen_features_on", false);
    * settings.setBool("bpros_features", false);
    
  Background subtracted game screen:
  
    * settings.setBool("screen_features_on", true);
    * settings.setBool("bpros_features", false);
    
  Bpros game screen:
 
    * settings.setBool("screen_features_on", false);
    * settings.setBool("bpros_features", true);

3. Set Bpros block size if Bpros features are used. In src/agents/IW1Search.cpp line 29 and src/agents/RolloutIW1Search.cpp line 21 ( m_bprosFeature = new BPROSFeature(5, 5); ), the first number is the height of a block, the second number is the width of a block. Default block size is set to (5, 5).

4. run IW1 and Rollout IW1.
The command to run IW1 is 

      ./ale -display_screen true -discount_factor 0.995 -randomize_successor_novelty true -max_sim_steps_per_frame 150000  -player_agent search_agent -search_method iw1  (ROM_PATH)
      
The command to run Rollout_IW(1) is 

      ./ale -display_screen true -discount_factor 0.995 -randomize_successor_novelty true -max_sim_steps_per_frame 150000  -player_agent search_agent -search_method rollout_iw1  (ROM_PATH)

Mind that Rollout_IW(1) can only work with Bpros features right now. 

and you have to substitute ROM_PATH for any of the games under *supported_roms* folder.

*max_sim_steps_per_frame* sets your budget, i.e. how many frames you can expand per lookahead. Each node is 5 frames of game play, so 150,000, is equivalent to 30,000 nodes generated. You don't need that many to find rewards, so you can use a smaller number to play the game much faster. 150,000 was the parameter chosen by Bellemare et al. The most expensive computation is calling the simulator to generate the successor state.

Right now, RolloutIW1 can only work with Manhattan BPros features. 
    
Credits
=======

We want to thank Marc Bellemare for making the ALE code available and the research group at U. Alberta.

The Classical Planning algorithms code is adapted from the Lightweight Automated Planning Toolkit (www.LAPKT.org)

