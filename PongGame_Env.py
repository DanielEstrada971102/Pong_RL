""" ****************************************************************************
* @file PongGame_Env.py                                                        *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief simple ping pong game created with pygame, this code is based on      *
*        malreddysid's available in:                                           *
*        https://github.com/llSourcell/pong_neural_network_live                *
*        and was adapted to be used as an OpenAI Gym enviroment (Env),         *
*        necessary to use the RL frameworks.                                    *
* @version 0.1                                                                 *
* @date 2021-12-10                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""

from gym import Env
from gym.spaces import Discrete, Box
import pygame
from numpy import random, array, float32

#RGB colors for our paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
#ACTIONS 
NOTMOVE  = 0
MOVEUP   = 1
MOVEDOWN = 2

#=========================Beging of the class definition =======================

class PongGame(Env):
    """
    This class build the ping pong game environment

    @parameters
    -----------
    -- game_speed : int

    @Attributes 
    -----------
    ->  WINDOW_WIDTH : int
        ... the width of the game window    
    ->  WINDOW_HEIGHT :int
        ... the height of the game window
    ->  PADDLE_WIDTH : int
        ... width of the paddles
    ->  PADDLE_HEIGHT : int
        ... height of the paddles
    ->  PADDLE_BUFFEER : int
        ... distance from the edge of the window
    ->  BALL_WIDTH : int
        ... width of the ball
    ->  BALL_HEIGHT : int
        ... height of the ball
    ->  PADDLE_1_POS : int
        ... human paddle position
    ->  PADDLE_2_POS : int
        ... enemy paddle position
    ->  BALL_X_POS : int
        ... x ball position
    ->  BALL_Y_POS : int
        ... y ball position
    ->  PADDLE_SPEDD : int
        ... speed of the paddles (vertical velocity)
    ->  BALL_X_SPEDD : int
        ... horizontal speed of the ball
    ->  BALL_Y_SPEDD : int
        ... vertical speed of the ball
    ->  GAME_SCORE : int
        ... score of the game. When human hits the ball, it recives +1 point
    ->  DONE : bool
        ... the game overs when the ball reaches the left side of the window
    ->  screen : pygame.display
        ... the game screen
    ->  action_space : gym.spaces.Discrete
        ... the action space,it mean, the type of action that can be taken
    ->  observation_space : gym.spaces.Box
        ... the observation space,it mean, the possible system states.

    """

    metadata = {'render.modes': ['human']}

    def __init__(self, game_speed=1):
        super(PongGame, self).__init__()

        self.WINDOW_WIDTH = 400
        self.WINDOW_HEIGHT = 400
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 60
        self.PADDLE_BUFFER = 10
        self.BALL_WIDTH = 10
        self.BALL_HEIGHT = 10
        self.PADDLE_SPEED = 0.8 * game_speed
        self.BALL_X_SPEED = game_speed
        self.BALL_Y_SPEED = 0.8 * game_speed
        
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, 
                                                      self.WINDOW_HEIGHT))

        # Defining action and observation space
        self.action_space = Discrete(3) # NOTMOVE, MOVEUP, MOVEDOWN
        # the object will be (paddle_1_pos, ball_X_pos, ball_Y_pos)
        low_ob = array([0, 0, 0])
        high_ob =array([self.WINDOW_HEIGHT - self.PADDLE_HEIGHT, 
                        self.WINDOW_WIDTH- self.BALL_WIDTH, 
                        self.WINDOW_HEIGHT - self.BALL_HEIGHT])
        self.observation_space = Box(low_ob, high_ob, shape=(3,), dtype=float32)


    #===========================================================================
    # In order to use this class as an enviroment for RL framewors 
    # it is necessary define three main methods
    # --> reset()
    # --> step(action)
    # --> (Optional) render(method='human')
    # and the last two attributes above --> action_space and observation_space

    def reset(self):
        """
        This function resets the environment to an initial state and returns an initial
        observation. In other words, it sets the initial configuration of the 
        game, it is called at the beginning of an episode.
        
        @Parameters
        -----------
        -
        @Retuns
        -------
        - observation (object): the initial observation.
        """
        # initial Ball position is chosen randomly
        self.BALL_X_POS =  self.WINDOW_WIDTH/2 - self.BALL_WIDTH/2
        self.BALL_Y_POS = random.randint(0,9)*(self.WINDOW_HEIGHT - 
                                                    self.BALL_HEIGHT)/9
        # initial Ball direction is chosen randomly
        self.BALL_X_DIR = random.choice([-1,1])
        self.BALL_Y_DIR = random.choice([-1,1])

        # initial Paddles positions are chosen in the middle 
        self.PADDLE_1_POS = self.WINDOW_HEIGHT / 2 - self.PADDLE_HEIGHT / 2
        self.PADDLE_2_POS = self.WINDOW_HEIGHT / 2 - self.PADDLE_HEIGHT / 2

        self.GAME_SCORE = 0
        self.GAME_OVER = 0

        state = [self.PADDLE_1_POS, self.BALL_X_POS, self.BALL_Y_POS]

        #important : the observation must be a numpy array
        return array(state).astype(float32)        


    def step(self, action):
        """
        This function performa one timestep of the environment's dynamics. 
        It is called to take an action with the environment, it accepts an 
        action and returns a tuple (observation, reward, done, info). 
        
        @Parameters
        -----------
        ->  action (object): an action provided by the agent or the player
        @Retuns
        -------
        ->  observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.updatePaddle1(action)
        self.updatePaddle2()
        self.updateBall()

        state = [self.PADDLE_1_POS, self.BALL_X_POS, self.BALL_Y_POS]

        done = bool(self.GAME_OVER == True)
        reward  = self.GAME_SCORE
        info = {}

        return array(state).astype(float32), reward, done, info


    def render(self, mode='human'):
        """
        This function allow to visualize the agent in action (the game). 
        
        @Parameters
        -----------
        --> mode ('human') : render to the current display or terminal and
            return nothing. 
        @Retuns
        -------
        -
        """
        pygame.event.pump()
        self.screen.fill(BLACK)

        pygame.font.init()
        font = pygame.font.SysFont('Consolas', 30)
        textScore = font.render('SCORE : {}'.format(self.GAME_SCORE), True, WHITE)
        self.screen.blit(textScore,(self.WINDOW_WIDTH/4,self.PADDLE_BUFFER))

        #update our paddle
        self.drawPaddle1()
        
        #update the paddleBot
        self.drawPaddle2()

        #draw the ball
        self.drawBall()
    
        #update the window
        pygame.display.flip()

    #===========================================================================
    #---The following functions define the behavior of the environment (game)---
    def updateBall(self):
        """
        This function implements the movement of the ball.

        """
        #update the x and y position
        self.BALL_X_POS = self.BALL_X_POS + self.BALL_X_DIR * self.BALL_X_SPEED
        self.BALL_Y_POS = self.BALL_Y_POS + self.BALL_Y_DIR * self.BALL_Y_SPEED

        #checks for a collision, if the ball hits the left side
        if (self.BALL_X_POS <= self.PADDLE_BUFFER + self.PADDLE_WIDTH and 
            self.BALL_Y_POS + self.BALL_HEIGHT >= self.PADDLE_1_POS and 
            self.BALL_Y_POS - self.BALL_HEIGHT <= self.PADDLE_1_POS + self.PADDLE_HEIGHT):
            #switches directions
            self.BALL_X_DIR = 1
            self.GAME_SCORE += 1
        
        #past it, the game over...
        elif (self.BALL_X_POS <= 0):
            self.BALL_X_DIR = 1
            self.GAME_OVER = 1 
        
        #check if hits the other side
        if (self.BALL_X_POS >= self.WINDOW_WIDTH - self.PADDLE_WIDTH - self.PADDLE_BUFFER and 
            self.BALL_Y_POS + self.BALL_HEIGHT >= self.PADDLE_2_POS and 
            self.BALL_Y_POS - self.BALL_HEIGHT <= self.PADDLE_2_POS + self.PADDLE_HEIGHT):
            #switch directions
            self.BALL_X_DIR = -1
            
        #past it
        elif (self.BALL_X_POS >= self.WINDOW_WIDTH - self.BALL_WIDTH):
            #positive score
            self.BALL_X_DIR = -1
            self.GAME_SCORE += 2

        else:
            pass
        
        #if it hits the top
        #move down
        if (self.BALL_Y_POS <= 0):
            self.BALL_Y_POS = 0
            self.BALL_Y_DIR = 1
        #if it hits the bottom, move up
        elif (self.BALL_Y_POS >= self.WINDOW_HEIGHT - self.BALL_HEIGHT):
            self.BALL_Y_POS = self.WINDOW_HEIGHT - self.BALL_HEIGHT
            self.BALL_Y_DIR = -1
        
        else:
            pass

    
    def updatePaddle1(self, action):
        """
        This function implements the movement of the human (or agent) paddle.
        
        """
        #if move up
        if (action == MOVEUP):
            self.PADDLE_1_POS = self.PADDLE_1_POS - self.PADDLE_SPEED
        #if move down
        if (action == MOVEDOWN):
            self.PADDLE_1_POS = self.PADDLE_1_POS + self.PADDLE_SPEED            
        if (action == NOTMOVE):
            self.PADDLE_1_POS = self.PADDLE_1_POS
        #don't let it move off the screen
        if (self.PADDLE_1_POS < 0):
            self.PADDLE_1_POS = 0
        if (self.PADDLE_1_POS > self.WINDOW_HEIGHT - self.PADDLE_HEIGHT):
            self.PADDLE_1_POS = self.WINDOW_HEIGHT - self.PADDLE_HEIGHT
        


    def updatePaddle2(self):
        """
        This function implements the movement of the boot (enemy) paddle.
        
        """
        #move down if ball is in upper half
        if (self.PADDLE_2_POS + self.PADDLE_HEIGHT/2 < self.BALL_Y_POS + self.BALL_HEIGHT/2):
            self.PADDLE_2_POS = self.PADDLE_2_POS + self.PADDLE_SPEED
        #move up if ball is in lower half
        if (self.PADDLE_2_POS + self.PADDLE_HEIGHT/2 > self.BALL_Y_POS + self.BALL_HEIGHT/2):
            self.PADDLE_2_POS = self.PADDLE_2_POS - self.PADDLE_SPEED
        #don't let it hit top
        if (self.PADDLE_2_POS < 0):
            self.PADDLE_2_POS = 0
        #dont let it hit bottom
        if (self.PADDLE_2_POS > self.WINDOW_HEIGHT - self.PADDLE_HEIGHT):
            self.PADDLE_2_POS = self.WINDOW_HEIGHT - self.PADDLE_HEIGHT
    #===========================================================================
    #-------- the below functions draw in the screen the game elements ---------
    def drawBall(self):
        #small rectangle, create it
        ball = pygame.Rect(self.BALL_X_POS, self.BALL_Y_POS, self.BALL_WIDTH, 
                            self.BALL_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, WHITE, ball)

    def drawPaddle1(self):
        #create it
        paddle1 = pygame.Rect(self.PADDLE_BUFFER, self.PADDLE_1_POS, 
                                self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, WHITE, paddle1)


    def drawPaddle2(self):
        #create it, opposite side
        paddle2 = pygame.Rect(self.WINDOW_WIDTH - self.PADDLE_BUFFER - self.PADDLE_WIDTH, 
                              self.PADDLE_2_POS, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        #draw it
        pygame.draw.rect(self.screen, WHITE, paddle2)


#==========================End of the class definition =========================

def main():
    game = PongGame()
    game.reset()
    game.render()
    PLAY = 1
    action = NOTMOVE

    while(1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                PLAY = 0
            elif event.type == pygame.KEYDOWN  and event.key == pygame.K_UP:
                action = MOVEUP
            elif event.type == pygame.KEYDOWN  and event.key == pygame.K_DOWN:
                action = MOVEDOWN
            else:
                action = NOTMOVE
        
        if game.GAME_OVER or not PLAY: break

        game.step(action)
        game.render()

if __name__=='__main__':
    main()