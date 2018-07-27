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
 *  common_constants.h
 *
 *  Defines a set of constants used by various parts of the player agent code
 *
 **************************************************************************** */

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <cassert>
#include <vector>
#include <valarray>
#include <cstdlib>
#include "../emucore/m6502/src/bspf/src/bspf.hxx"


// Define actions
enum Action {
    PLAYER_A_NOOP           = 0,
    PLAYER_A_FIRE           = 1,
    PLAYER_A_UP             = 2,
    PLAYER_A_RIGHT          = 3,
    PLAYER_A_LEFT           = 4,
    PLAYER_A_DOWN           = 5,
    PLAYER_A_UPRIGHT        = 6,
    PLAYER_A_UPLEFT         = 7,
    PLAYER_A_DOWNRIGHT      = 8,
    PLAYER_A_DOWNLEFT       = 9,
    PLAYER_A_UPFIRE         = 10,
    PLAYER_A_RIGHTFIRE      = 11,
    PLAYER_A_LEFTFIRE       = 12,
    PLAYER_A_DOWNFIRE       = 13,
    PLAYER_A_UPRIGHTFIRE    = 14,
    PLAYER_A_UPLEFTFIRE     = 15,
    PLAYER_A_DOWNRIGHTFIRE  = 16,
    PLAYER_A_DOWNLEFTFIRE   = 17,
    PLAYER_B_NOOP           = 18,
    PLAYER_B_FIRE           = 19,
    PLAYER_B_UP             = 20,
    PLAYER_B_RIGHT          = 21,
    PLAYER_B_LEFT           = 22,
    PLAYER_B_DOWN           = 23,
    PLAYER_B_UPRIGHT        = 24,
    PLAYER_B_UPLEFT         = 25,
    PLAYER_B_DOWNRIGHT      = 26,
    PLAYER_B_DOWNLEFT       = 27,
    PLAYER_B_UPFIRE         = 28,
    PLAYER_B_RIGHTFIRE      = 29,
    PLAYER_B_LEFTFIRE       = 30,
    PLAYER_B_DOWNFIRE       = 31,
    PLAYER_B_UPRIGHTFIRE    = 32,
    PLAYER_B_UPLEFTFIRE     = 33,
    PLAYER_B_DOWNRIGHTFIRE  = 34,
    PLAYER_B_DOWNLEFTFIRE   = 35,
    RESET                   = 40, // MGB: Use SYSTEM_RESET to reset the environment. 
    UNDEFINED               = 41,
    RANDOM                  = 42,
    SAVE_STATE              = 43,
    LOAD_STATE              = 44,
    SYSTEM_RESET            = 45,
    LAST_ACTION_INDEX       = 50
};

#define PLAYER_A_MAX (18)
#define PLAYER_B_MAX (36)

string action_to_string(Action a);

//  Define datatypes
typedef vector<int>         IntVect;
typedef vector<Action>      ActionVect;
typedef vector<double>      FloatVect;
typedef vector<IntVect>     IntMatrix;
typedef vector<FloatVect>   FloatMatrix;

// reward type for RL interface
typedef int reward_t;
// return (sum of discounted rewards) type for RL
typedef float return_t;

// Other constant values
#define RAM_LENGTH 128
#define CUSTOM_PALETTE_SIZE 1020    // Number of colors in custom palette
#define BLACK_COLOR_IND 1000        // color index in custom palette for Black
#define RED_COLOR_IND 1001          // color index in custom palette for Red
#define WHITE_COLOR_IND 1003        // color index in custom palette for White
#define SECAM_COLOR_IND 1010        // starting index in the custom palette for
                                    // the eight SECAM colors
// Added by xhy, used in IW1Search.cpp
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 210
#define SCREEN_SIZE (210*160)

#define NUM_COLORS 256

#define T 8
#define K 5
#define NUM_STEP 3

#define HIDDEN1_SIZE 2048
#define HIDDEN2_SIZE 2048 

#endif // __CONSTANTS_H__

