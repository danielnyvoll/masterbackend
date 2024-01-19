import pygame
from pygame.rect import Rect
import numpy as np
from math import sin, cos, fabs, acos, pi, inf
from values import *

class Environment:
    """
    Represents the environment of simulation.
    """
    def __init__(self, window):
        self.window = window
        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.list_centers = None
        self.list_radius = None
        self.list_rotation = None

    def draw(self, params):
        """
        This method call all other methods for drawing.

        :param params: params for drawing the window.
        """
        self.update(params)
        self.draw_field()
        self.draw_ball()
        self.draw_players()
        self.draw_soccer_goal_and_scoreboard()
    
    def draw_ball(self):
        # draw ball
        center = self.list_centers[0]
        # Drawing player's inner circle
        pygame.draw.circle(self.window, WHITE_COLOR, (center[0], center[1]), 
            self.list_radius[1], 0)
        
    def draw_players(self):
        """
        Drawing players and ball.

        :param params: params for drawing the window.
        """
        for i in range(1, len(self.list_centers)):
            center = self.list_centers[i]
            final_position = self.list_radius[i] * np.array([cos(self.list_rotation[i]), 
                sin(self.list_rotation[i])]) + center
            if i <= len(self.list_centers)/2:
                color = RED_COLOR
            else:
                color = YELLOW_COLOR
            # Drawing player's inner circle
            pygame.draw.circle(self.window, color, (center[0], center[1]), 
                self.list_radius[i], 0)
            # Drawing player's outer circle
            pygame.draw.circle(self.window, GRAY_COLOR, (center[0], center[1]), 
                self.list_radius[i], 4)
            # Drawing player's orientation
            pygame.draw.line(self.window, GRAY_COLOR, (center[0], center[1]), 
                (final_position[0], final_position[1]), 3)

      

    def draw_field(self):
        """
        Drawing soccer field.

        :param window: pygame's window where the drawing will occur.
        """
        self.window.fill((35,142,35))
        

        pygame.draw.circle(self.window, (255,255,255), (round(SCREEN_WIDTH/2), 
            round(SCREEN_HEIGHT/2)), 70, 3)
        pygame.draw.line(self.window, (255,255,255), (round(SCREEN_WIDTH/2), 30), 
            (round(SCREEN_WIDTH/2), SCREEN_HEIGHT - 30), 3)
        pygame.draw.line(self.window, (255,255,255), (30, 30), 
            (round(SCREEN_WIDTH)-30, 30), 3)
        pygame.draw.line(self.window, (255,255,255), (30, 30), 
            (30, round(SCREEN_HEIGHT)-30), 3)
        pygame.draw.line(self.window, (255,255,255), (round(SCREEN_WIDTH)-30, 30), 
            (round(SCREEN_WIDTH)-30, round(SCREEN_HEIGHT)-30), 3)
        pygame.draw.line(self.window, (255,255,255), (30, round(SCREEN_HEIGHT)-30), 
            (round(SCREEN_WIDTH)-30, round(SCREEN_HEIGHT)-30), 3)

       
    def draw_soccer_goal_and_scoreboard(self):
        """
        Drawing soccer goal and scoreboard.
        """
        scoreboard="Left " + str(self.left_goal) + " x " + str(self.right_goal) + " Right"
        textsurface = self.font.render(scoreboard, False, WHITE_COLOR)
        # Drawing soccer goal
        pygame.draw.rect(self.window, (0, 0, 0), 
            Rect(0, round(SCREEN_HEIGHT)/2-100, 30, 200))
        pygame.draw.rect(self.window, (0, 0, 0), 
            Rect(round(SCREEN_WIDTH)-30, round(SCREEN_HEIGHT)/2-100, 30, 200))
        # scoreboard
        pygame.draw.rect(self.window, (0, 0, 0), 
            Rect(28, round(SCREEN_HEIGHT-30), 250, 30))

        self.window.blit(textsurface, (40,round(SCREEN_HEIGHT-30)))

        
    def update(self, params):
        """
        Update params of environment.

        :param params: params for drawing the window.
        """

        self.window = params["window"]
        self.list_centers = params["list_centers"]
        self.list_radius = params["list_radius"]
        self.list_rotation = params["list_rotation"]
        self.left_goal = params["left_goal"]
        self.right_goal = params["right_goal"]