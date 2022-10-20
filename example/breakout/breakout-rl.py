# SOURCE : https://www.101computing.net/breakout-tutorial-using-pygame-getting-started/

#Import the pygame library and initialise the game engine
import pygame
#Let's import the Paddle Class & the Ball Class
from paddle import Paddle
from ball import Ball
from brick import Brick

pygame.init()

# Define some colors
WHITE = (255,255,255)
DARKBLUE = (36,90,190)
LIGHTBLUE = (0,176,240)
RED = (255,0,0)
ORANGE = (255,100,0)
YELLOW = (255,255,0)
GREEN = (0, 255, 0)
BRIGHT_BLUE = (0, 255, 255)
COLORS = [RED, ORANGE, YELLOW, GREEN, BRIGHT_BLUE]

PADDLE_WIDTH = 100
PADDLE_HEIGHT = 10
PADDLE_INITIAL_POSITION_X = 350
PADDLE_INITIAL_POSITION_Y = 700
BALL_DIMENSION = 10
BALL_INITIAL_POSITION_X = 345
BALL_INITIAL_POSITION_Y = 330

BALL_INITIAL_POSITION_X = 345
BALL_INITIAL_POSITION_Y = 330

BRICK_WIDTH = 50
BRICK_HEIGHT = 10
BRICK_N_COLUMN = 10
BRICK_N_ROW = 20
BRICK_MARGIN = 60
BRICK_MARGIN_COLUMN = 20
BRICK_MARGIN_ROW = 5

score = 0
lives = 3

# Open a new window
size = (800, 800)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Breakout Game")

#This will be a list that will contain all the sprites we intend to use in our game.
all_sprites_list = pygame.sprite.Group()

#Create the Paddle
paddle = Paddle(LIGHTBLUE, PADDLE_WIDTH, PADDLE_HEIGHT)
paddle.rect.x = PADDLE_INITIAL_POSITION_X
paddle.rect.y = PADDLE_INITIAL_POSITION_Y

#Create the ball sprite
ball = Ball(WHITE,BALL_DIMENSION,BALL_DIMENSION)
ball.rect.x = BALL_INITIAL_POSITION_X
ball.rect.y = BALL_INITIAL_POSITION_Y

all_bricks = pygame.sprite.Group()

for j in range(BRICK_N_ROW):
    for i in range(BRICK_N_COLUMN):
        brick = Brick(COLORS[j % len(COLORS)],BRICK_WIDTH,BRICK_HEIGHT)
        brick.rect.x = BRICK_MARGIN + i* (BRICK_WIDTH + BRICK_MARGIN_COLUMN)
        brick.rect.y = BRICK_MARGIN + j* (BRICK_HEIGHT + BRICK_MARGIN_ROW)
        all_sprites_list.add(brick)
        all_bricks.add(brick)

# Add the paddle and the ball to the list of sprites
all_sprites_list.add(paddle)
all_sprites_list.add(ball)

# The loop will carry on until the user exits the game (e.g. clicks the close button).
carryOn = True

# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

frame = 0
# -------- Main Program Loop -----------
while carryOn:
    # To make sure it doesnt do endless loop
    if ball.velocity[1] == 0:
        ball.velocity[1] = 1
    """
    if paddle.rect.x < ball.rect.x:
        paddle.moveRight(5)
    else:
        paddle.moveLeft(5)
    """
    # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
              carryOn = False # Flag that we are done so we exit this loop
    #Moving the paddle when the use uses the arrow keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        paddle.moveLeft(5)
    if keys[pygame.K_RIGHT]:
        paddle.moveRight(5)

    # --- Game logic should go here
    all_sprites_list.update()

    #Check if the ball is bouncing against any of the 4 walls:
    if ball.rect.x>=800:
        ball.velocity[0] = -ball.velocity[0]
    if ball.rect.x<=0:
        ball.velocity[0] = -ball.velocity[0]
    if ball.rect.y > PADDLE_INITIAL_POSITION_Y + 50:
        ball.velocity[1] = -ball.velocity[1]
        lives -= 1

        # reset position when loses a live
        ball.rect.x = BALL_INITIAL_POSITION_X
        ball.rect.y = BALL_INITIAL_POSITION_Y

        if lives == 0:
            #Display Game Over Message for 3 seconds
            font = pygame.font.Font(None, 74)
            text = font.render("GAME OVER", 1, WHITE)
            screen.blit(text, (250,300))
            pygame.display.flip()
            pygame.time.wait(3000)

            #Stop the Game
            carryOn=False

    if ball.rect.y<40:
        ball.velocity[1] = -ball.velocity[1]

    #Detect collisions between the ball and the paddles
    if pygame.sprite.collide_mask(ball, paddle):
      ball.rect.x -= ball.velocity[0]
      ball.rect.y -= ball.velocity[1]
      ball.bounce()

    #Check if there is the ball collides with any of bricks
    brick_collision_list = pygame.sprite.spritecollide(ball,all_bricks,False)
    for brick in brick_collision_list:
      ball.bounce()
      score += 1
      brick.kill()
      if len(all_bricks)==0:
           #Display Level Complete Message for 3 seconds
            font = pygame.font.Font(None, 74)
            text = font.render("LEVEL COMPLETE", 1, WHITE)
            screen.blit(text, (200,300))
            pygame.display.flip()
            pygame.time.wait(3000)

            #Stop the Game
            carryOn=False

    # --- Drawing code should go here
    # First, clear the screen to dark blue.
    screen.fill(DARKBLUE)
    pygame.draw.line(screen, WHITE, [0, 38], [800, 38], 2)

    #Display the score and the number of lives at the top of the screen
    font = pygame.font.Font(None, 34)
    meta_font = pygame.font.Font(None, 18)
    text = font.render("Score: " + str(score), 1, WHITE)
    screen.blit(text, (20,10))
    text = font.render("Lives: " + str(lives), 1, WHITE)
    screen.blit(text, (650,10))
    text = meta_font.render("Pad : {}, {}".format(paddle.rect.x, paddle.rect.y), 5, WHITE)
    screen.blit(text, (200,10))
    text = meta_font.render("Ball v : {}, {}".format(ball.velocity[0], ball.velocity[1]), 1, WHITE)
    screen.blit(text, (300,10))
    text = meta_font.render("Ball : {}, {}".format(ball.rect.x, ball.rect.y), 1, WHITE)
    screen.blit(text, (400,10))
    frame += 1

    #Now let's draw all the sprites in one go. (For now we only have 2 sprites!)
    all_sprites_list.draw(screen)

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

#Once we have exited the main program loop we can stop the game engine:
pygame.quit()