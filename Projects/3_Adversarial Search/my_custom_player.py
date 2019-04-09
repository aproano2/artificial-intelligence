import random
from sample_players import DataPlayer
from isolation.isolation import _WIDTH, _HEIGHT

FLAG = 4
# Define coordinated of the center points of the board
# CENTER_COORDS[0] contains the X-axis
# CENTER_COORDS[1] contains the Y-axis
CENTER_COORDS = list()            
for v in [_WIDTH, _HEIGHT]:
    if v % 2:
        CENTER_COORDS.append([v // 2])
    else:
        CENTER_COORDS.append([v // 2, v // 2 - 1])


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)
        
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # iterative deepening
            depth_limit = 9
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta_search(state, depth)
                self.queue.put(best_move)
            
    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
    
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), alpha, beta, depth)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move


    def min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)
    
        if depth <= 0:
            return self.utility(state)
    
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


    def max_value(self, state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)
    
        if depth <= 0:
            if FLAG == 0:
                return self.utility(state)
            elif FLAG == 1:
                return self.heuristic(state) 
            elif FLAG == 2:
                return self.heuristic2(state)
            elif FLAG == 3:
                return self.heuristic3(state)
            elif FLAG == 4:
                return self.heuristic4(state)
    
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def utility(self, state):
        player_loc = state.locs[self.player_id]
        player_liberties = state.liberties(player_loc)
        opponent_loc = state.locs[1 - self.player_id]
        opponent_liberties = state.liberties(opponent_loc)
        return len(player_liberties) - len(opponent_liberties)
    
    def heuristic(self, state):
        """
        This heuristic takes into account the position of the 
        player. If the player tries to keep itself around the center
        of the board, isolation becomes more challenging. 
        """
        player_loc = state.locs[self.player_id]
        player_liberties = state.liberties(player_loc)
        player_distance = distance_to_center(player_loc)
        opponent_loc = state.locs[1 - self.player_id]
        opponent_liberties = state.liberties(opponent_loc)
        opponent_distance = distance_to_center(opponent_loc)
        
        liberties_diff = len(player_liberties) - len(opponent_liberties)
        distance_diff = opponent_distance - player_distance
        
        return liberties_diff + distance_diff
    
    def heuristic2(self, state):
        """
        This heuristic takes into account the position of the 
        player. If the player tries to keep itself around the center
        of the board, isolation becomes more challenging. 
        """
        player_loc = state.locs[self.player_id]
        player_liberties = state.liberties(player_loc)
        player_distance = distance_to_center(player_loc)
        opponent_loc = state.locs[1 - self.player_id]
        opponent_liberties = state.liberties(opponent_loc)
        opponent_distance = distance_to_center(opponent_loc)
        
        liberties_diff = len(player_liberties) - len(opponent_liberties)
        distance_diff = opponent_distance - player_distance
        
        return liberties_diff / max(len(player_liberties), len(opponent_liberties)) + \
               distance_diff / max(opponent_distance, player_distance)
        
    def heuristic3(self, state):
        """
        This heuristic takes into account the position of the 
        player. If the player tries to keep itself around the center
        of the board, isolation becomes more challenging. 
        """
        player_loc = state.locs[self.player_id]
        player_liberties = state.liberties(player_loc)
        player_distance = distance_to_center(player_loc)
        opponent_loc = state.locs[1 - self.player_id]
        opponent_liberties = state.liberties(opponent_loc)
        opponent_distance = distance_to_center(opponent_loc)
        
        liberties_diff = len(player_liberties) - len(opponent_liberties)
        distance_diff = opponent_distance - player_distance
        
        if distance_diff > 0:
            metric = 2*len(player_liberties) - len(opponent_liberties)
        elif distance_diff < 0:
            metric = len(player_liberties) - 2*len(opponent_liberties)
        else:
            metric = len(player_liberties) - len(opponent_liberties)
        
        return metric 

    
    def heuristic4(self, state):
        """
        This heuristic takes into account the position of the 
        player. If the player tries to keep itself around the center
        of the board, isolation becomes more challenging. 
        """
        player_loc = state.locs[self.player_id]
        player_liberties = state.liberties(player_loc)
        player_distance = distance_to_center(player_loc)
        opponent_loc = state.locs[1 - self.player_id]
        opponent_liberties = state.liberties(opponent_loc)
        
        max_distance = distance_to_center(0)
        
        liberties_diff = len(player_liberties) - len(opponent_liberties)

        # Penalize moves that go more than half-way from the center of the board
        if player_distance >= max_distance / 2:
            metric = len(player_liberties) - 2*len(opponent_liberties)
        else:
            metric = len(player_liberties) - len(opponent_liberties)        
        return metric 

    
def distance_to_center(player_loc):
    # (x,y) coordinates of the player
    player_coord = player_loc % (_WIDTH + 2), player_loc // (_WIDTH + 2)                              
    # Get the distance to the center point(s)
    distance = [abs(c_x - player_coord[0]) + abs(c_y - player_coord[1])
                for c_x in CENTER_COORDS[0] for c_y in CENTER_COORDS[1]]
    return min(distance)
