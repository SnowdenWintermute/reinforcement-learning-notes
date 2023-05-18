import pygame, sys, time, random
from pygame.surfarray import array3d

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)

class SnakeEnv():

    def __init__(self, frame_size_x, frame_size_y):
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))

        self.reset()

    def reset(self):
        self.game_window.fill(BLACK)
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self.spawn_food()
        self.food_exists = True
        self.direction = 'RIGHT'
        self.action = self.direction
        self.score = 0
        self.steps = 0
        print("GAME RESET")

    def change_direction(self, action, direction):
        if action == "UP" and direction != "DOWN":
            direction = "UP"
        if action == "DOWN" and direction != "UP":
            direction = "DOWN"
        if action == "RIGHT" and direction != "LEFT":
            direction = "RIGHT"
        if action == "LEFT" and direction != "RIGHT":
            direction = "LEFT"
        return direction

    def move(self, direction, snake_pos):
        if direction == "UP":
            snake_pos[1] -= 10
        if direction == "DOWN":
            snake_pos[1] += 10
        if direction == "LEFT":
            snake_pos[0] -= 10
        if direction == "RIGHT":
            snake_pos[0] += 10

        return snake_pos

    def spawn_food(self):
        return [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10] 

    def eat(self):
        condition_1 = True if self.food_pos[0] in range(self.snake_pos[0]-10, self.snake_pos[0]+10) else False
        condition_2 = True if self.food_pos[1] in range(self.snake_pos[1]-10, self.snake_pos[1]+10) else False
        return condition_1 and condition_2

    def human_step(self, event):
        action = None
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = "UP"
            if event.key == pygame.K_DOWN:
                action = "DOWN"
            if event.key == pygame.K_LEFT:
                action = "LEFT"
            if event.key == pygame.K_RIGHT:
                action = "RIGHT"
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        return action

    def display_score(self, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render("Score: " + str(self.score), True, color)

        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def game_is_over(self):
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x - 10:
            self.end_game()
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y - 10:
            self.end_game()
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.end_game()

    def end_game(self):
        message = pygame.font.SysFont('arial', 45)
        message_surface = message.render("GAME ENDED", True, RED)
        message_rect = message_surface.get_rect()
        message_rect.midtop = (self.frame_size_x/2, self.frame_size_y/4)
        self.game_window.fill(BLACK)
        self.game_window.blit(message_surface, message_rect)
        self.display_score(RED, 'times', 20)
        pygame.display.flip()

        time.sleep(3)
        pygame.quit()
        sys.exit()
        pass

snake_env = SnakeEnv(600, 600)
difficulty = 10
fps_controller = pygame.time.Clock()
check_errors = pygame.init()

pygame.display.set_caption('Snake Game')

while True:
    for event in pygame.event.get():
        snake_env.action = snake_env.human_step(event)

    snake_env.direction = snake_env.change_direction(snake_env.action, snake_env.direction)
    snake_env.move(snake_env.direction, snake_env.snake_pos)
    snake_env.snake_body.insert(0, list(snake_env.snake_pos))
    if snake_env.eat():
        snake_env.score += 1
        snake_env.food_exists = False
    else:
        snake_env.snake_body.pop()
    if not snake_env.food_exists:
        snake_env.food_pos = snake_env.spawn_food()
        snake_env.food_exists = True

    snake_env.game_window.fill(BLACK)
    for pos in snake_env.snake_body:
        pygame.draw.rect(snake_env.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))

    pygame.draw.rect(snake_env.game_window, GREEN, pygame.Rect(snake_env.food_pos[0], snake_env.food_pos[1], 10, 10))
    snake_env.game_is_over()

    snake_env.display_score(WHITE, 'times', 20)
    pygame.display.update()

    fps_controller.tick(33)

    img = array3d(snake_env.game_window)

