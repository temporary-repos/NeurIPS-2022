import pygame
import pymunk.pygame_util
import numpy as np
import os
import copy


class BallBox:
    def __init__(self, dt=0.2, res=(32, 32), init_pos=(3, 3), init_std=0, wall=None, gravity=(0.0, 0.0)):
        pygame.init()

        self.dt = dt
        self.res = res
        if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
            pygame.display.set_mode(res, 0, 24)
            self.screen = pygame.Surface(res, pygame.SRCCOLORKEY, 24)
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, res[0], res[1]), 0)
        else:
            self.screen = pygame.display.set_mode(res, 0, 24)
        self.gravity = gravity
        self.initial_position = init_pos
        self.initial_std = init_std
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = None

        self.dd = 0

    def _clear(self):
        self.screen.fill(pygame.color.THECOLORS["black"])

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        position = np.array(self.initial_position) + self.initial_std * np.random.normal(size=(2,))
        position = np.clip(position, self.dd + radius +1, self.res[0]-self.dd-radius-1)
        position = position.tolist()
        body.position = position

        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1.0
        shape.color = pygame.color.THECOLORS["white"]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def run(self, iterations=20, sequences=500, angle_limits=(0, 360), velocity_limits=(10, 25), radius=3,
            flip_gravity=None, save=None, filepath='../../data/balls.npz', delay=None):
        if save:
            images = np.empty((sequences, iterations, self.res[0], self.res[1]), dtype=np.float32)
            state = np.empty((sequences, iterations, 4), dtype=np.float32)

        dd = self.dd
        self.static_lines = [pymunk.Segment(self.space.static_body, (dd, dd), (dd, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (dd, dd), (self.res[0]-dd, dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (dd, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (self.res[0]-dd, dd), 0.0)]
        for line in self.static_lines:
            line.elasticity = 1.0
            # line.color = pygame.color.THECOLORS["white"]
        # self.space.add(self.static_lines)

        for sl in self.static_lines:
            self.space.add(sl)

        for s in range(sequences):

            if s % 100 == 0:
                print(s)

            angle = np.random.uniform(*angle_limits)
            velocity = np.random.uniform(*velocity_limits)
            # controls[:, s] = np.array([angle, velocity])
            ball = self.fire(angle, velocity, radius)
            for i in range(iterations):
                self._clear()
                self.space.debug_draw(self.draw_options)
                self.space.step(self.dt)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == 'png':
                    pygame.image.save(self.screen, os.path.join(filepath, "bouncing_balls_%02d_%02d.png" % (s, i)))
                elif save == 'npz':
                    images[s, i] = pygame.surfarray.array2d(self.screen).swapaxes(1, 0).astype(np.float32) / (2**24 - 1)
                    state[s, i] = list(ball.body.position) + list(ball.body.velocity)

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)

        # if save == 'npz':
        #     np.savez(os.path.abspath(filepath), images=images, state=state)
        return images, state


if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    scale = 1

    # Create data dir
    if not os.path.exists('../bouncing_ball/'):
        os.makedirs('../bouncing_ball/')

    # Get class labels
    sequences = 3000
    num_gravs = 16

    # Sample gravities
    gs = [[np.cos(k * np.pi / (num_gravs // 2)), np.sin(k * np.pi / (num_gravs // 2))] for k in range(num_gravs)]

    # Set up files
    train_images, train_states, train_labels = [], [], []
    valid_images, valid_states, valid_labels = [], [], []
    target_images, target_states, target_labels = [], [], []
    np.random.seed(1234)

    selected = np.random.choice(num_gravs, num_gravs, replace=False)
    train_selected = np.sort(selected[:10])
    valid_selected = np.sort(selected[10:12])
    target_selected = np.sort(selected[12:])

    # Over each grav, sample
    for idx, g in enumerate(gs):
        g_range = 3 + np.random.random_sample()
        g_x, g_y = np.array(g) * g_range
        print(g_range)
        print(g_x, g_y)
        cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=8, wall=None,
                            gravity=(g_x, g_y))
        i, s = cannon.run(delay=None, iterations=100, sequences=sequences, radius=3*scale, angle_limits=(0, 360),
                            velocity_limits=(10.0*scale, 10.0*scale), save='npz')
        if idx in train_selected:
            train_images.append(i)
            train_states.append(s)
            train_labels.append(np.full(sequences, idx))
        elif idx in valid_selected:
            valid_images.append(i)
            valid_states.append(s)
            valid_labels.append(np.full(sequences, idx))
        elif idx in target_selected:
            target_images.append(i)
            target_states.append(s)
            target_labels.append(np.full(sequences, idx))
        else:
            raise NotImplemented

    train_images = np.concatenate(train_images, axis=0)
    train_states = np.concatenate(train_states, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    valid_images = np.concatenate(valid_images, axis=0)
    valid_states = np.concatenate(valid_states, axis=0)
    valid_labels = np.concatenate(valid_labels, axis=0)
    target_images = np.concatenate(target_images, axis=0)
    target_states = np.concatenate(target_states, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)

    # def movie_to_frame(images):
    #     """ Compiles a list of images into one composite frame """
    #     n_steps, w, h = images.shape
    #     colors = np.linspace(0.4, 1, n_steps)
    #     image = np.zeros((w, h))
    #     for i, color in zip(images, colors):
    #         image = np.clip(image + i * color, 0, color)
    #     return image
    
    # import matplotlib.pyplot as plt
    # np.random.seed(5678)
    # selected_idx = np.random.choice(train_size, 10, replace=False)
    # for idx in selected_idx:
    #     plt.imshow(movie_to_frame(train_images[idx]), cmap='gray')
    #     plt.savefig('../../experiments/examples/source_{}.png'.format(idx))
    
    # selected_idx = np.random.choice(test_size, 10, replace=False)
    # for idx in selected_idx:
    #     plt.imshow(movie_to_frame(test_images[idx]), cmap='gray')
    #     plt.savefig('../../experiments/examples/target_{}.png'.format(idx))

    test_images, test_states, test_labels = [], [], []
    np.random.seed(3456)
    train_len = train_images.shape[0]
    p = np.random.permutation(train_len)
    train_images = train_images[p]
    train_states = train_states[p]
    train_labels = train_labels[p]

    test_images.append(copy.deepcopy(train_images[train_len // 2:]))
    test_states.append(copy.deepcopy(train_states[train_len // 2:]))
    test_labels.append(copy.deepcopy(train_labels[train_len // 2:]))

    train_images = copy.deepcopy(train_images[:train_len // 2])
    train_states = copy.deepcopy(train_states[:train_len // 2])
    train_labels = copy.deepcopy(train_labels[:train_len // 2])

    valid_len = valid_images.shape[0]
    p = np.random.permutation(valid_len)
    valid_images = valid_images[p]
    valid_states = valid_states[p]
    valid_labels = valid_labels[p]

    test_images.append(copy.deepcopy(valid_images[valid_len // 2:]))
    test_states.append(copy.deepcopy(valid_states[valid_len // 2:]))
    test_labels.append(copy.deepcopy(valid_labels[valid_len // 2:]))

    valid_images = copy.deepcopy(valid_images[:valid_len // 2])
    valid_states = copy.deepcopy(valid_states[:valid_len // 2])
    valid_labels = copy.deepcopy(valid_labels[:valid_len // 2])

    test_images = np.concatenate(test_images, axis=0)
    test_states = np.concatenate(test_states, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_len = test_images.shape[0]
    p = np.random.permutation(test_len)
    test_images = test_images[p]
    test_states = test_states[p]
    test_labels = test_labels[p]

    np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_train.npz"), images=train_images, state=train_states, label=train_labels)

    np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_valid.npz"), images=valid_images, state=valid_states, label=valid_labels)

    for label in np.unique(test_labels):
        label_idx = np.where(test_labels == label)[0]
        num_sample = label_idx.shape[0]
        eval_idx, pred_idx = label_idx[:100], label_idx[100:]
        eval_images_i = test_images[eval_idx]
        eval_states_i = test_states[eval_idx]
        eval_labels_i = test_labels[eval_idx]
        np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_eval_{}.npz".format(label)), images=eval_images_i, state=eval_states_i, label=eval_labels_i)

        pred_images_i = test_images[pred_idx]
        pred_states_i = test_states[pred_idx]
        pred_labels_i = test_labels[pred_idx]
        np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_pred_{}.npz".format(label)), images=pred_images_i, state=pred_states_i, label=pred_labels_i)

    target_len = target_images.shape[0]
    p = np.random.permutation(target_len)
    target_images = target_images[p]
    target_states = target_states[p]
    target_labels = target_labels[p]

    for label in np.unique(target_labels):
        label_idx = np.where(target_labels == label)[0]
        num_sample = label_idx.shape[0]
        eval_idx, pred_idx = label_idx[:num_sample // 2], label_idx[num_sample // 2:]
        eval_images_i = target_images[eval_idx]
        eval_states_i = target_states[eval_idx]
        eval_labels_i = target_labels[eval_idx]
        np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_target_eval_{}.npz".format(label)), images=eval_images_i, state=eval_states_i, label=eval_labels_i)

        pred_images_i = target_images[pred_idx]
        pred_states_i = target_states[pred_idx]
        pred_labels_i = target_labels[pred_idx]
        np.savez(os.path.abspath("../bouncing_ball/mixed_gravity_16_target_pred_{}.npz".format(label)), images=pred_images_i, state=pred_states_i, label=pred_labels_i)

    print('Done')
