import time


class fpsCalculator:
    def __init__(self):
        self.updated_step = False
        self.fps_current = 0.0
        self.render_fps_counter = 0
        self.render_fps_timer = time.time()

    def update(self):
        self.render_fps_counter += 1
        current_time = time.time()
        time_diff = current_time - self.render_fps_timer
        if time_diff >= 1: # Update once per second
            self.updated_step = True
            self.fps_current = self.render_fps_counter
            self.render_fps_counter = 0
            self.render_fps_timer = current_time

            # total_entities = len(self.players) + len(self.platforms) + len(self.ability_generated_objects)
            # print(F"FPS: {int(self.fps_current)}, total_steps: {self.current_step_cpu[:5]}, player scores: {self.episode_total_rewards}")
            # print(F"FPS: {int(self.fps_current)}, total_steps: {self.current_step_cpu[:5]}")
            # print(F"FPS: {int(self.fps_current)}")
        else:
            self.updated_step = False

    def get_fps(self):
        return self.fps_current




