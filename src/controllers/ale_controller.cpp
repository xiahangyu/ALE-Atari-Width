/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  controller.hpp 
 *
 *  Superclass defining a variety of controllers -- main loops interfacing with
 *   an agent in a particular way. This superclass handles work common to all 
 *   controllers, e.g. loading ROM settings and constructing the environment
 *   wrapper.
 **************************************************************************** */

#include "ale_controller.hpp"
#include "../games/Roms.hpp"

#include "../common/display_screen.h"

ALEController::ALEController(OSystem* osystem):
  m_osystem(osystem),
  m_settings(buildRomRLWrapper(m_osystem->settings().getString("rom_file"))),
  m_environment(m_osystem, m_settings.get()) {

  if (m_settings.get() == NULL) {
    std::cerr << "Unsupported ROM file: " << std::endl;
    exit(1);
  }
  else {
    m_environment.reset();
  }
}

void ALEController::display() {
  DisplayScreen* display = m_osystem->p_display_screen;

  if (display) // Display the screen if applicable
    display->display_screen(m_osystem->console().mediaSource());
    //display->display_screen(m_environment.getScreen());
}

/** Added by xhy, for counting background histogram*/
void ALEController::count_bghist(){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->count_bghist();
}

/** Added by xhy, get background matrix from histogram*/
void ALEController::count_bgMatrix(){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->count_bgMatrix();
}

/** xhy, save bg as png*/
void ALEController::save_bg(const string& filename){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->save_bg(filename);
}

/** xhy, save gb as matrix*/
void ALEController::saveBgAsMatrix(const string& filename){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->saveBgAsMatrix(filename);
}

/** xhy, load background from matrix file*/
void ALEController::loadBgFromMatrix(const string& filename){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->loadBgFromMatrix(filename);
}

/** Added by xhy, for saving screens. */
void ALEController::save_screen(const string& filename) {
  DisplayScreen* display = m_osystem->p_display_screen;

  if (display) // Display the screen if applicable
    display->save_screen(m_osystem->console().mediaSource(), filename);
}

void ALEController::saveScreenAsMatrix(const string& filename){
  DisplayScreen* display = m_osystem->p_display_screen;

  if(display)
    display->saveScreenAsMatrix(m_osystem->console().mediaSource(), filename);
}

reward_t ALEController::applyActions(Action player_a, Action player_b) {
  reward_t sum_rewards = 0;
  // Perform different operations based on the first player's action 
  switch (player_a) {
    case LOAD_STATE: // Load system state
      // Note - this does not reset the game screen; so that the subsequent screen
      //  is incorrect (in fact, two screens, due to colour averaging)
      m_environment.load();
      break;
    case SAVE_STATE: // Save system state
      m_environment.save();
      break;
    case SYSTEM_RESET:
      m_environment.reset();
      break;
    default:
      // Pass action to emulator!
	// static RomSettings*  new_settings = buildRomRLWrapper(m_osystem->settings().getString("rom_file"));
	// static StellaEnvironment test(m_osystem,  new_settings );
	// ALEState new_state = m_environment.cloneState();
	// //test.reset();
	// test.restoreState( new_state );
	// reward_t test_rewards = test.act( player_a, player_b);
	sum_rewards = m_environment.act(player_a, player_b);
	// if (sum_rewards != 0 )
	//     std::cout << "reward: " << sum_rewards << "test-reward: " << test_rewards << std::endl;
	// assert( test_rewards == sum_rewards );
	// if (sum_rewards != 0 )
	//     std::cout << "reward: " << sum_rewards << std::endl;

      break;
  }
  return sum_rewards;
}

