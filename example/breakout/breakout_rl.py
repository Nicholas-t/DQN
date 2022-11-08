# SOURCE : https://www.101computing.net/breakout-tutorial-using-pygame-getting-started/

#Import the pygame library and initialise the game engine
import pygame
from random import randint
from pygame.locals import (
    K_LEFT,
    K_RIGHT
)
#Let's import the Paddle Class & the Ball Class
from example.breakout.paddle import Paddle
from example.breakout.ball import Ball
from example.breakout.brick import Brick

class Game:
    def __init__( self, mode='human',  lives=3, framerate = 60):
        pygame.init()
        # Define some colors
        self.WHITE = (255,255,255)
        self.DARKBLUE = (36,90,190)
        self.LIGHTBLUE = (0,176,240)
        self.RED = (255,0,0)
        self.ORANGE = (255,100,0)
        self.YELLOW = (255,255,0)
        self.GREEN = (0, 255, 0)
        self.BRIGHT_BLUE = (0, 255, 255)
        self.COLORS = [
            self.RED,
            self.ORANGE,
            self.YELLOW,
            self.GREEN,
            self.BRIGHT_BLUE
        ]

        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 10
        self.PADDLE_INITIAL_POSITION_X = 350
        self.PADDLE_INITIAL_POSITION_Y = 700
        self.BALL_DIMENSION = 10
        self.BALL_INITIAL_POSITION_X = 345
        self.BALL_INITIAL_POSITION_Y = 700

        self.BRICK_WIDTH = 50
        self.BRICK_HEIGHT = 10
        self.BRICK_N_COLUMN = 10
        self.BRICK_N_ROW = 20
        self.BRICK_MARGIN = 60
        self.BRICK_MARGIN_COLUMN = 20
        self.BRICK_MARGIN_ROW = 5

        self.score = 0
        self.lives = lives
        self.mode = mode
        self.framerate = framerate
        if self.mode == 'human':
            self.screen_mode = pygame.SHOWN
        else:
            self.screen_mode = pygame.HIDDEN

        # Open a new window
        self.size = (800, 800)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Breakout Game")

        #This will be a list that will contain all the sprites we intend to use in our game.
        self.all_sprites_list = pygame.sprite.Group()

        #Create the Paddle
        self.paddle = Paddle(self.LIGHTBLUE, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.paddle.rect.x = self.PADDLE_INITIAL_POSITION_X
        self.paddle.rect.y = self.PADDLE_INITIAL_POSITION_Y
        self.all_sprites_list.add(self.paddle)

        #Create the ball sprite
        self.ball = Ball(self.WHITE,self.BALL_DIMENSION, self.BALL_DIMENSION)
        self.ball.rect.x = self.BALL_INITIAL_POSITION_X
        self.ball.rect.y = self.BALL_INITIAL_POSITION_Y
        self.all_sprites_list.add(self.ball)

        self.all_bricks = pygame.sprite.Group()
        for j in range(self.BRICK_N_ROW):
            for i in range(self.BRICK_N_COLUMN):
                brick = Brick(self.COLORS[j % len(self.COLORS)],self.BRICK_WIDTH,self.BRICK_HEIGHT)
                brick.rect.x = self.BRICK_MARGIN + i* (self.BRICK_WIDTH + self.BRICK_MARGIN_COLUMN)
                brick.rect.y = self.BRICK_MARGIN + j* (self.BRICK_HEIGHT + self.BRICK_MARGIN_ROW)
                self.all_sprites_list.add(brick)
                self.all_bricks.add(brick)

        # The clock will be used to control how fast the screen updates
        self.clock = pygame.time.Clock()

    def get_action(self, pressed_keys):
        right = pressed_keys[K_RIGHT]
        left = pressed_keys[K_LEFT]
        n_pressed = sum([right, left])
        if n_pressed != 1:
            action = 0
        elif n_pressed == 1 and right:
            action = 1
        elif n_pressed == 1 and left:
            action = 2
        else:
            action = 0
        return action

    def step_frame(self, action):
        if action == 1:
            self.paddle.moveRight(5)
        if action == 2:
            self.paddle.moveLeft(5)
        # To make sure it doesnt do endless loop
        if self.ball.velocity[1] == 0:
            self.ball.velocity[1] = 1
        # --- Game logic should go here
        self.all_sprites_list.update()
        #Check if the ball is bouncing against any of the 4 walls:
        if self.ball.rect.x>=800:
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.x<=0:
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.y>790:
            self.ball.velocity[1] = -self.ball.velocity[1]
            self.lives -= 1
            self.ball.rect.x = self.BALL_INITIAL_POSITION_X
            self.ball.rect.y = self.BALL_INITIAL_POSITION_Y
            self.ball.velocity = [randint(4,8),randint(-8,8)]

            if self.lives == 0:
                #Display Game Over Message for 3 seconds
                font = pygame.font.Font(None, 74)
                text = font.render("GAME OVER", 1, self.WHITE)
                self.screen.blit(text, (250,300))
                pygame.display.flip()
                pygame.time.wait(3000)
                #Stop the Game
                self.carryOn=False
        if self.ball.rect.y<40:
            self.ball.velocity[1] = -self.ball.velocity[1]
        #Detect collisions between the ball and the paddles
        if pygame.sprite.collide_mask(self.ball, self.paddle):
            self.ball.rect.x -= self.ball.velocity[0]
            self.ball.rect.y -= self.ball.velocity[1]
            self.ball.bounce()
        #Check if there is the ball collides with any of bricks
        brick_collision_list = pygame.sprite.spritecollide(self.ball, self.all_bricks,False)
        for brick in brick_collision_list:
            self.ball.bounce()
            self.score += 1
            brick.kill()
            if len(self.all_bricks)==0:
                #Display Level Complete Message for 3 seconds
                    font = pygame.font.Font(None, 74)
                    text = font.render("LEVEL COMPLETE", 1, self.WHITE)
                    self.screen.blit(text, (200,300))
                    pygame.display.flip()
                    pygame.time.wait(3000)
                    #Stop the Game
                    self.carryOn=False
        self.update_screen()

        
    def turn_on_screen(self):
        self.screen = pygame.display.set_mode(
            self.size, 
            flags=pygame.SHOWN
        )
        self.include_info=True
    
    def update_screen(self):
        # --- Drawing code should go here
        # First, clear the screen to dark blue.
        self.screen.fill(self.DARKBLUE)
        pygame.draw.line(self.screen, self.WHITE, [0, 38], [800, 38], 2)
        #Display the score and the number of lives at the top of the screen
        font = pygame.font.Font(None, 34)
        text = font.render("Score: " + str(self.score), 1, self.WHITE)
        self.screen.blit(text, (20,10))
        text = font.render("Lives: " + str(self.lives), 1, self.WHITE)
        self.screen.blit(text, (650,10))
        #Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
        self.all_sprites_list.draw(self.screen)

    def render_screen(self):
        pygame.display.flip()

    def play(self):
        # The loop will carry on until the user exits the game (e.g. clicks the close button).
        self.carryOn = True
        # -------- Main Program Loop -----------
        while self.carryOn:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self.carryOn = False # Flag that we are done so we exit this loop
            #Moving the paddle when the use uses the arrow keys
            pressed_keys = pygame.key.get_pressed()
            action = self.get_action(pressed_keys)
            self.step_frame(action)
            self.render_screen()
            # --- Limit to 60 frames per second
            self.clock.tick(60)
