from values import *
from tools import *
from math import sin, cos, sqrt, pi, inf, acos

class Sensors:
    """
    Represents the sensors of a player.
    """
    def __init__(self, agent):
        self.flag_points = self.init_flag_points()
        self.agent_center = agent.pose
        self.full_vision = None

    def set_full_vision(self, full_vision):
        self.full_vision = full_vision

    def init_flag_points(self):
        """
        Find the flags around the field.

        return: a list of points.
        rtype: list
        """
        points = []
        for i in range(11):
            points.append((round(SCREEN_WIDTH * i/10), 0))
        for i in range(1,11):
            points.append((SCREEN_WIDTH, round(SCREEN_HEIGHT * i/10)))
        for i in range(10):
            points.append((round(SCREEN_WIDTH * i/10), SCREEN_HEIGHT))
        for i in range(1,10):
            points.append((0, round(SCREEN_HEIGHT * i/10)))
        
        

        return points

    def calculate_distance(self, agent, list_centers):
        """
        Calculate the vector distance between agent and other players, ball and flags.

        param agent: the agent that we are calculating the distances.
        type agent: Player
        param list_centers: the list of center's position that players and ball.
        type list_centers: Pose.position
        return: list of distance to points
        rtype: list
        """

        self.agent_center = agent.pose
        points = self.flag_points + list_centers
        dirvector_list = []
        for point in points:
            center = Vector2(self.agent_center.position.x * M2PIX,
                self.agent_center.position.y * M2PIX)
            dirvector = Vector2(*point).dirvector(center)
            dirvector = self.is_visible(dirvector)
            dirvector_list.append(dirvector)
        
        return dirvector_list

    def is_visible(self, vector):
        """
        Checks if a point is visible for a agent.

        param vector: vector that links the center of the agent to the point in question.
        type vector: Vector2
        return: the same vector if is visible and infinity vector if isn't.
        rtype: Vector2
        """
        if not self.full_vision:
            vector_agent = TransformCartesian(1, self.agent_center.rotation)
            vector_agent = Vector2(vector_agent.x, vector_agent.y)
            angle = acos(vector_agent.dot(vector)/vector.magnitude())

            if angle <= pi/4:
                return vector

            return Vector2(inf, inf)
        
        return vector
        