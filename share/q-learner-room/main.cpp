#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using std::cout;
using std::endl;

#define DEBUG 0

void init_array(std::vector<std::vector<double> > & array, int rows, int cols, double value)
{
    for (int r = 0; r < rows; r++) {
        std::vector<double> vec;
        for (int c = 0; c < cols; c++) {
            vec.push_back(value);
        }
        array.push_back(vec);
    }
}

void print_array(std::vector<std::vector<double> > & array)
{
    for (auto it_r = array.begin(); it_r != array.end(); it_r++) {
        for (auto it_c = it_r->begin(); it_c != it_r->end(); it_c++) {
            cout << *it_c << "\t";
        }
        cout << endl;
    }
}

void normalize_array(std::vector<std::vector<double> > & array)
{
    // Find largest value:
    double champ = -std::numeric_limits<double>::infinity();
    for (auto it_r = array.begin(); it_r != array.end(); it_r++) {
        for (auto it_c = it_r->begin(); it_c != it_r->end(); it_c++) {
            if (*it_c > champ) {
                champ = *it_c;
            }
        }
    }

    // Normalize all values:    
    for (auto it_r = array.begin(); it_r != array.end(); it_r++) {
        for (auto it_c = it_r->begin(); it_c != it_r->end(); it_c++) {
            *it_c /= champ;
        }
    }
}

const int INVALID = -1;
const int R_GOAL = 100;
const int GOAL_STATE = 5;

int max_action(std::vector<std::vector<double> > & Q,
               std::vector<std::vector<double> > & R,
               int state)
{
    double champ_value = -std::numeric_limits<double>::infinity();
    int champ_action = -1;
    int a = 0;
    for (auto it = Q[state].begin(); it != Q[state].end(); it++) {
        if (R[state][a] != INVALID && *it > champ_value) { // TODO: Better invalid
            champ_value = *it;
            champ_action = a;
        }
        a++;
    }
    return champ_action;
}

int main(int argc, char *argv[])
{

    int num_states = 6;  // rows
    int num_actions = 6; // cols
    double gamma = 0.8;

    int num_episodes = 1e5;

    // Random number generator
    std::default_random_engine gen;
    gen.seed(std::chrono::system_clock::now().time_since_epoch().count());

    // Random state selector
    std::uniform_int_distribution<int> rand_state(0,num_states-1);

    // Random action selector
    std::uniform_int_distribution<int> rand_action(0,num_actions-1);

    std::vector<std::vector<double> > Q; // Q table
    init_array(Q, num_states, num_actions, 0);

    std::vector<std::vector<double> > R; // State Rewards
    init_array(R, num_states, num_actions, 0);

    cout << "Initialized Q Array..." << endl;
    print_array(Q);

    R[0][0] = R[0][1] = R[0][2] = R[0][3] = R[0][5] = INVALID;
    R[1][0] = R[1][1] = R[1][2] = R[1][4] = INVALID;
    R[2][0] = R[2][1] = R[2][2] = R[2][4] = R[2][5] = INVALID;
    R[3][0] = R[3][3] = R[3][5] = INVALID;
    R[4][1] = R[4][2] = R[4][4] = INVALID;
    R[5][0] = R[5][2] = R[5][3] = INVALID;

    R[1][5] = R[4][5] = R[5][5] = R_GOAL;

    cout << "Initialized R Array... " << endl;
    print_array(R);

    for (int n = 0; n < num_episodes; n++) {
        // Select random initial state (TODO);
        int state = rand_state(gen);

        while(true) {
            // Select random action from current state
            // Action is selecting next state
            int action;
            do {
                action = rand_action(gen);
            } while (R[state][action] == INVALID); // TODO: CLEANER

            int next_state = action; // The next state is the action

            // Update Q-table
            Q[state][action] = R[state][action] + gamma * Q[next_state][max_action(Q, R, next_state)];

            // Are we at the GOAL State?
            if (state == GOAL_STATE) {
                break;
            }

            // Go to next state:
            state = next_state;
        }
    }
    cout << "Q Table: " << endl;
    print_array(Q);

    cout << "Normalized Q Table: " << endl;
    normalize_array(Q);
    print_array(Q);
}
