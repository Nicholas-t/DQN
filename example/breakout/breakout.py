# SOURCE : https://www.101computing.net/breakout-tutorial-using-pygame-getting-started/

#Import the pygame library and initialise the game engine
import pygame
from pygame.locals import (
    K_LEFT,
    K_RIGHT
)
#Let's import the Paddle Class & the Ball Class
from paddle import Paddle
from ball import Ball
from brick import Brick

class Game:
    def __init__(
        self,
        mode='human',
        lives=3
    ):
        pygame.init()
        # Define some colors
        self.WHITE = (255,255,255)
        self.DARKBLUE = (36,90,190)
        self.LIGHTBLUE = (0,176,240)
        self.RED = (255,0,0)
        self.ORANGE = (255,100,0)
        self.YELLOW = (255,255,0)
        self.score = 0
        self.lives = lives
        self.mode = mode
        if self.mode == 'human':
            self.screen_mode = pygame.SHOWN
        else:
            self.screen_mode = pygame.HIDDEN

        # Open a new window
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Breakout Game")

        #This will be a list that will contain all the sprites we intend to use in our game.
        self.all_sprites_list = pygame.sprite.Group()

        #Create the Paddle
        self.paddle = Paddle(self.LIGHTBLUE, 100, 10)
        self.paddle.rect.x = 350
        self.paddle.rect.y = 560
        self.all_sprites_list.add(self.paddle)

        #Create the ball sprite
        self.ball = Ball(self.WHITE,10,10)
        self.ball.rect.x = 345
        self.ball.rect.y = 195
        self.all_sprites_list.add(self.ball)

        self.all_bricks = pygame.sprite.Group()
        for i in range(7):
            brick = Brick(self.RED,80,30)
            brick.rect.x = 60 + i* 100
            brick.rect.y = 60
            self.all_sprites_list.add(brick)
            self.all_bricks.add(brick)
        for i in range(7):
            brick = Brick(self.ORANGE,80,30)
            brick.rect.x = 60 + i* 100
            brick.rect.y = 100
            self.all_sprites_list.add(brick)
            self.all_bricks.add(brick)
        for i in range(7):
            brick = Brick(self.YELLOW,80,30)
            brick.rect.x = 60 + i* 100
            brick.rect.y = 140
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
        # --- Game logic should go here
        self.all_sprites_list.update()
        #Check if the ball is bouncing against any of the 4 walls:
        if self.ball.rect.x>=790:
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.x<=0:
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.y>590:
            self.ball.velocity[1] = -self.ball.velocity[1]
            self.lives -= 1
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

breakout = Game()
breakout.play()