import pygame
import math
import time
import numpy as np

class KalmanFilter:
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F  # State Transition matrix
        self.B = B  # Control matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Error covariance matrix
        self.x = x0  # Initial state estimate
        self.I = np.eye(self.n)  # Identity matrix

    def predict(self, u=None):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)
        
class Missile:
    def __init__(self, radar, initial_position, speed):
        self.radar = radar
        self.position = np.array(initial_position, dtype=float)
        self.speed = speed  # missile speed
        self.navigation_constant = .5  # N, tunable parameter
        self.trail = []  # to store the missile's path, now includes time
        self.active = True
        self.lock_lost_time = None
        self.max_turn_rate = np.radians(2)  # max turn rate in radians per update

        # Initial speed direction
        self.direction = np.array([0, -1], dtype=float)

        # Kalman filter for LOS rate estimation
        self.kalman_filter = KalmanFilter(
            F=np.eye(2),  # State Transition matrix (2x2 identity matrix)
            B=None,  # Control matrix (not used in this case)
            H=np.eye(2),  # Observation matrix (2x2 identity matrix)
            Q=np.eye(2) * 0.01,  # Process noise covariance (2x2 diagonal matrix with small values)
            R=np.eye(2) * 0.1,  # Measurement noise covariance (2x2 diagonal matrix with larger values)
            P=np.eye(2),  # Error covariance matrix (2x2 identity matrix)
            x0=np.zeros(2)  # Initial state estimate (2D zero vector)
        )

    def calculate_LOS_rate(self):
        target_position = self.radar.track_object
        if not target_position:
            return None

        # Use Kalman filter to estimate target position
        measured_position = np.array(target_position, dtype=float)
        self.kalman_filter.predict()
        self.kalman_filter.update(measured_position)
        estimated_position = self.kalman_filter.x

        dx = estimated_position[0] - self.position[0]
        dy = estimated_position[1] - self.position[1]
        distance = math.hypot(dx, dy)
        return dx / distance, dy / distance

    def update(self):
        LOS_rate = self.calculate_LOS_rate()
        if LOS_rate:
            LOS_dx, LOS_dy = LOS_rate
            LOS_direction = np.array([LOS_dx, LOS_dy], dtype=float)

            # Compute the angle between the current direction and LOS direction
            cos_angle = np.dot(self.direction, LOS_direction)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # to avoid errors due to numerical precision
            angle = np.arccos(cos_angle)

            # Compute the rotation direction (-1 for left, 1 for right)
            rotation_direction = np.sign(np.cross(self.direction, LOS_direction))

            # Compute the angle to rotate at this update
            delta_angle = np.minimum(self.max_turn_rate, angle)

            # Create the rotation matrix
            rotation_matrix = np.array([[np.cos(delta_angle), -rotation_direction * np.sin(delta_angle)],
                                        [rotation_direction * np.sin(delta_angle), np.cos(delta_angle)]])

            # Rotate the direction
            self.direction = np.dot(rotation_matrix, self.direction)

            # Update position
            self.position += self.speed * self.direction
            self.trail.append((tuple(self.position), time.time(), 255))  # Initial alpha value is set to 255 (fully opaque)

            if self.radar.track_object and math.hypot(self.position[0] - self.radar.track_object[0],
                                                      self.position[1] - self.radar.track_object[1]) < 20:
                print("BOOM")
                self.active = False
        else:
            if self.lock_lost_time is None:
                self.lock_lost_time = time.time()
            elif time.time() - self.lock_lost_time > 5:
                self.active = False

    def draw(self, screen):
        if not self.active:
            # Draw explosion
            explosion_radius = 30
            if self.lock_lost_time is not None:
                time_difference = time.time() - self.lock_lost_time
                alpha = max(0, 255 - int(time_difference * 255 / 2))
            else:
                alpha = 255
            color = (*self.radar.WHITE, alpha)
            pygame.draw.circle(screen, color, (int(self.position[0]), int(self.position[1])), explosion_radius)
        else:
            # Calculate the rotation angle based on the missile's direction
            rotation_angle = np.degrees(np.arctan2(-self.direction[1], self.direction[0]))
            rotated_triangle = pygame.transform.rotate(self.radar.triangle_surface, rotation_angle)
            rotated_triangle_rect = rotated_triangle.get_rect(center=(self.position[0], self.position[1]))
    
            # Draw the rotated triangle
            screen.blit(rotated_triangle, rotated_triangle_rect)
    
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                # Calculate alpha based on age
                age = time.time() - self.trail[i][1]
                alpha = max(0, 255 - int(age * 255 / 5))  # Trail lasts for 5 seconds, adjust the division value as desired
                color = (*self.radar.WHITE, alpha)
                pygame.draw.line(screen, color, self.trail[i - 1][0], self.trail[i][0])


                
    def calculate_time_until_impact(self):
        target_position = self.radar.track_object
        if target_position:
            dx = target_position[0] - self.position[0]
            dy = target_position[1] - self.position[1]
            distance = math.hypot(dx, dy)
            return distance / self.speed
        return None

class Radar:
    def __init__(self, width, height, radius, sweep_speed, angle_increment, track_angle_range, track_speed,
                 track_sweep_speed, max_detect_count, max_detect_age):
        self.width = width
        self.height = height
        self.radius = radius
        self.sweep_speed = sweep_speed
        self.angle_increment = angle_increment
        self.track_angle_range = track_angle_range
        self.track_speed = track_speed
        self.track_sweep_speed = track_sweep_speed
        self.max_detect_count = max_detect_count
        self.max_detect_age = max_detect_age
        self.max_detection_range = radius  # Set the maximum detection range to 200 (adjust as needed)


        # Initialize Pygame
        pygame.init()

        # Set up the display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Search and Track Radar")

        # Colors
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255, 255)
        self.YELLOW = (255, 255, 0)
        self.WHITE = (255, 255, 255)


        # Retro font settings
        self.FONT_SIZE = 16
        self.font = pygame.font.Font(None, self.FONT_SIZE)

        # Calculate the center of the radar
        self.center_x = self.width // 2
        self.center_y = self.height - self.radius

        # Initialize current angle
        self.current_angle = 0

        # Initialize tracking angle
        self.track_angle = 0

        # Initialize track sweep angle
        self.track_sweep_angle = -self.track_angle_range / 2

        # Initialize active object
        self.active_object = None

        # Initialize tracking object
        self.track_object = None
        self.track_history = []

        # Initialize tracking data
        self.track_data = {"bearing": 0, "range": 0, "velocity": 0}

        # Create a surface for the detections
        self.detections_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Create a list to store the search detections
        self.search_detections = []

        # Initialize velocity graph variables
        self.velocity_graph_x = []
        self.velocity_graph_y = []

        # Initialize Kalman filter
        self.kalman_filter = KalmanFilter(
            F=np.eye(2),  # State Transition matrix (2x2 identity matrix)
            B=None,  # Control matrix (not used in this case)
            H=np.array([[1, 0], [0, 1]]),  # Observation matrix (2x2 identity matrix)
            Q=np.eye(2) * 0.01,  # Process noise covariance (2x2 diagonal matrix with small values)
            R=np.eye(2) * 0.1,  # Measurement noise covariance (2x2 diagonal matrix with larger values)
            P=np.eye(2),  # Error covariance matrix (2x2 identity matrix)
            x0=np.zeros(2)  # Initial state estimate (2D zero vector)
        )
        self.missile = None  # No active missile at start
        self.missiles = []  # No active missiles at start
        self.triangle_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.polygon(self.triangle_surface, self.WHITE, [(0, -10), (-10, 10), (10, 10)])

        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get the mouse position when clicked
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Create an active object
                self.active_object = (mouse_x, mouse_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                # Release the active object
                self.active_object = None
            elif event.type == pygame.MOUSEMOTION:
                # Move the active object
                if self.active_object:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.active_object = (mouse_x, mouse_y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m and self.active_object is not None:
                    self.missiles.append(Missile(self, [self.center_x, self.center_y], 2))
        return True

    def update_angle(self):
        self.current_angle += self.angle_increment

    def update_tracking_angle(self):
        if self.track_object is None and self.search_detections:
            # Find the detection with the highest count (most recent) from the search radar
            latest_search_detection = max(self.search_detections, key=lambda d: d["count"])
            target_angle = latest_search_detection["angle"]
        elif self.track_object:
            # Calculate the angle to the tracked object
            dx = self.track_object[0] - self.center_x
            dy = self.center_y - self.track_object[1]
            target_angle = math.degrees(math.atan2(dy, dx))
        else:
            return
    
        # Adjust the tracking angle towards the target angle
        if abs(target_angle - self.track_angle) > 180:
            if target_angle > self.track_angle:
                self.track_angle -= self.track_speed
            else:
                self.track_angle += self.track_speed
        else:
            if target_angle > self.track_angle:
                self.track_angle += self.track_speed
            else:
                self.track_angle -= self.track_speed
    
        self.track_angle %= 360


    def update_track_sweep_angle(self):
        if self.track_sweep_angle >= self.track_angle_range:
            self.track_sweep_direction = -1
        elif self.track_sweep_angle <= 0:
            self.track_sweep_direction = 1
        self.track_sweep_angle += self.track_sweep_speed * self.track_sweep_direction

    def draw_radar_circle(self):
        pygame.draw.arc(self.screen, self.GREEN,
                        (self.center_x - self.radius, self.center_y - self.radius, self.radius * 2, self.radius * 2),
                        math.radians(0), math.radians(180), 3)

    def draw_radar_sweep_line(self):
        end_x = self.center_x + int(self.radius * math.cos(math.radians(self.current_angle)))
        end_y = self.center_y - int(self.radius * math.sin(math.radians(self.current_angle)))
        pygame.draw.line(self.screen, self.GREEN, (self.center_x, self.center_y), (end_x, end_y), 2)

    def draw_tracking_radar(self):
        track_start_x = self.center_x + int(
            self.radius * math.cos(math.radians(self.track_angle - self.track_angle_range / 2)))
        track_start_y = self.center_y - int(
            self.radius * math.sin(math.radians(self.track_angle - self.track_angle_range / 2)))
        track_end_x = self.center_x + int(
            self.radius * math.cos(math.radians(self.track_angle + self.track_angle_range / 2)))
        track_end_y = self.center_y - int(
            self.radius * math.sin(math.radians(self.track_angle + self.track_angle_range / 2)))
        pygame.draw.line(self.screen, self.YELLOW, (self.center_x, self.center_y), (track_start_x, track_start_y), 3)
        pygame.draw.line(self.screen, self.YELLOW, (self.center_x, self.center_y), (track_end_x, track_end_y), 3)

    def draw_track_sweep_line(self):
        track_sweep_x = self.center_x + int(
            self.radius * math.cos(math.radians(self.track_angle + self.track_sweep_angle - self.track_angle_range / 2)))
        track_sweep_y = self.center_y - int(
            self.radius * math.sin(math.radians(self.track_angle + self.track_sweep_angle - self.track_angle_range / 2)))
        pygame.draw.line(self.screen, self.YELLOW, (self.center_x, self.center_y), (track_sweep_x, track_sweep_y), 2)

    def check_object_detection(self):
        detection = None
        if self.active_object:
            obj_x, obj_y = self.active_object
            distance_to_object = math.hypot(obj_x - self.center_x, self.center_y - obj_y)
            angle_to_object = math.degrees(math.atan2(self.center_y - obj_y, obj_x - self.center_x))
            if angle_to_object < 0:
                angle_to_object += 360
    
            # Check if the object is within the radar's effective range
            if 0 < distance_to_object <= self.max_detection_range:
                if (
                    (
                        self.current_angle - self.angle_increment < angle_to_object <= self.current_angle or
                        self.current_angle <= angle_to_object < self.current_angle - self.angle_increment
                    ) or (
                        self.current_angle - self.angle_increment < angle_to_object <= 0 and
                        self.current_angle + self.angle_increment > 360
                    ) or (
                        self.track_angle - self.track_angle_range / 2 <= angle_to_object <= self.track_angle + self.track_angle_range / 2
                    )
                ):
                    radar_type = "search"
                    if (
                        self.track_angle - self.track_angle_range / 2 <= angle_to_object <= self.track_angle + self.track_angle_range / 2
                    ):
                        radar_type = "track"
                    detection = {
                        "position": self.active_object,
                        "distance": distance_to_object,
                        "angle": angle_to_object,
                        "time": time.time(),
                        "count": 1,
                        "radar": radar_type,
                    }
                    
                    # Update track_object even if no new detection is made
                    if self.track_object is not None:
                        self.track_object = self.active_object
                        self.track_history.append({"position": self.track_object, "time": time.time()})
                    elif (
                        self.track_angle - self.track_angle_range / 2 <= angle_to_object <= self.track_angle + self.track_angle_range / 2
                    ):
                        self.track_object = self.active_object
                        self.track_history.append({"position": self.track_object, "time": time.time()})
        return detection


    def update_search_detections(self, current_time):
        self.search_detections = [
            d for d in self.search_detections if current_time - d["time"] <= self.max_detect_age
        ]

    def update_track_history(self, current_time):
        self.track_history = [
            history for history in self.track_history if current_time - history["time"] <= self.max_detect_age
        ]

    def calculate_tracking_data(self):
        if len(self.track_history) >= 2:
            self.track_data["bearing"] = self.track_angle
            self.track_data["range"] = math.hypot(
                self.track_object[0] - self.center_x, self.center_y - self.track_object[1]
            )
            dx = self.track_history[-1]["position"][0] - self.track_history[0]["position"][0]
            dy = self.track_history[-1]["position"][1] - self.track_history[0]["position"][1]
            dt = self.track_history[-1]["time"] - self.track_history[0]["time"]

            if dt != 0:
                self.track_data["velocity"] = math.hypot(dx / dt, dy / dt)
            else:
                self.track_data["velocity"] = 0

            # Use Kalman filter to estimate velocity
            measured_velocity = np.array([[dx / dt], [dy / dt]])
            self.kalman_filter.predict()
            self.kalman_filter.update(measured_velocity)
            estimated_velocity = self.kalman_filter.x
            self.track_data["velocity"] = np.linalg.norm(estimated_velocity)

            # Add velocity data to the graph
            self.velocity_graph_x.append(time.time())
            self.velocity_graph_y.append(self.track_data["velocity"])

    def draw_detections(self):
        self.detections_surface.fill((0, 0, 0, 0))
        current_time = time.time()
        for detection in self.search_detections:
            if detection["radar"] == "search":
                alpha = 255 - int((current_time - detection["time"]) / self.max_detect_age * 255)
                pygame.draw.circle(self.detections_surface, (*self.BLUE[:3], alpha), detection["position"], 5)
        self.screen.blit(self.detections_surface, (0, 0))

    def draw_active_object(self):
        if self.active_object:
            pygame.draw.circle(self.screen, self.RED, self.active_object, 10)

    def draw_tracking_information(self):
        text_surface = self.font.render(f"Bearing: {self.track_data['bearing']:.1f}", True, self.YELLOW)
        self.screen.blit(text_surface, (self.width - 200, 20))
        text_surface = self.font.render(f"Range: {self.track_data['range']:.1f}", True, self.YELLOW)
        self.screen.blit(text_surface, (self.width - 200, 40))
        text_surface = self.font.render(f"Velocity: {self.track_data['velocity']:.1f}", True, self.YELLOW)
        self.screen.blit(text_surface, (self.width - 200, 60))

        # Display search radar detections
        y_offset = 80
        for detection in self.search_detections:
            if detection["radar"] == "search":
                text_surface = self.font.render(f"Search Detection: Bearing - {detection['angle']:.1f}, Range - {detection['distance']:.1f}",
                                                True, self.YELLOW)
                self.screen.blit(text_surface, (self.width - 200, y_offset))
                y_offset += 20
                
        # Calculate and display time until impact for each missile
        missile_times = []
        for missile in self.missiles:
            time_until_impact = missile.calculate_time_until_impact()
            if time_until_impact is not None:
                missile_times.append(f"Time until impact: {time_until_impact:.2f}ms")
        for i, missile_time in enumerate(missile_times):
            text_surface = self.font.render(missile_time, True, self.YELLOW)
            self.screen.blit(text_surface, (20, 20 + i * 20))


    def draw_predicted_path(self):
        if len(self.track_history) >= 2:
            predicted_path = []
            dt = self.track_history[-1]["time"] - self.track_history[0]["time"]
            predicted_steps = int(dt / self.sweep_speed)
    
            for i in range(predicted_steps):
                t = self.track_history[-1]["time"] + i * self.sweep_speed
                x = self.kalman_filter.x[0, 0] * (t - self.track_history[0]["time"]) + self.track_history[0]["position"][0]
                y = self.kalman_filter.x[1, 0] * (t - self.track_history[0]["time"]) + self.track_history[0]["position"][1]
                predicted_path.append((int(x), int(y)))
    
            if len(predicted_path) >= 2:
                pygame.draw.lines(self.screen, self.GREEN, False, predicted_path, 3)
                pygame.draw.circle(self.screen, self.GREEN, predicted_path[-1], 4)

    def draw_velocity_graph(self):
        graph_width = 500
        graph_height = 300
        graph_x = self.width - graph_width - 220  # Adjust the x-coordinate of the graph's starting position
        graph_y = 30  # Adjust the y-coordinate of the graph's starting position

        # Draw graph border
        pygame.draw.rect(self.screen, self.YELLOW, (graph_x, graph_y, graph_width, graph_height), 3)

        if len(self.velocity_graph_x) >= 2:
            current_time = time.time()

            # Calculate the maximum time value within the sliding window
            max_time = current_time - self.max_detect_age

            # Find the index of the first velocity data point within the sliding window
            start_index = next((i for i, t in enumerate(self.velocity_graph_x) if t >= max_time), None)

            if start_index is not None:
                # Truncate the velocity data to include only the points within the sliding window
                truncated_velocity_graph_x = self.velocity_graph_x[start_index:]
                truncated_velocity_graph_y = self.velocity_graph_y[start_index:]

                # Calculate min and max velocity values for scaling within the sliding window
                min_velocity = min(truncated_velocity_graph_y)
                max_velocity = max(truncated_velocity_graph_y)
                velocity_range = max_velocity - min_velocity

                if velocity_range != 0:
                    # Calculate the x and y scaling factors for the graph
                    x_scale = graph_width / self.max_detect_age
                    y_scale = graph_height / velocity_range

                    # Create a list of points to draw the graph within the sliding window
                    points = []
                    for i in range(len(truncated_velocity_graph_x)):
                        x = int(graph_x + (truncated_velocity_graph_x[i] - current_time + self.max_detect_age) * x_scale)
                        y = int(graph_y + graph_height - (truncated_velocity_graph_y[i] - min_velocity) * y_scale)
                        points.append((x, y))

                    # Draw the graph
                    pygame.draw.lines(self.screen, self.YELLOW, False, points)

                    # Draw graph title
                    title_surface = self.font.render("Velocity Graph", True, self.YELLOW)
                    title_rect = title_surface.get_rect(center=(graph_x + graph_width / 2, graph_y - 20))
                    self.screen.blit(title_surface, title_rect)

                    # Draw grid lines
                    grid_spacing = graph_width / 4
                    for i in range(1, 4):
                        x = int(graph_x + i * grid_spacing)
                        pygame.draw.line(self.screen, self.YELLOW, (x, graph_y), (x, graph_y + graph_height), 1)
                    for i in range(1, 3):
                        y = int(graph_y + i * (graph_height / 2))
                        pygame.draw.line(self.screen, self.YELLOW, (graph_x, y), (graph_x + graph_width, y), 1)

            else:
                # Display "TRACK LOST" when the tracked target is lost
                flash_time = int(time.time() * 2) % 2  # Flashing effect using time
                if flash_time == 0:
                    text_surface = self.font.render("TRACK LOST", True, self.YELLOW)
                    text_rect = text_surface.get_rect(center=(graph_x + graph_width / 2, graph_y + graph_height / 2))
                    self.screen.blit(text_surface, text_rect)                    
                
    def update_display(self):
        pygame.display.flip()

    def delay(self):
        pygame.time.delay(self.sweep_speed)

    def run(self):
        # Game loop
        running = True
        while running:
            running = self.handle_events()
    
            self.update_angle()
            self.update_tracking_angle()
            self.update_track_sweep_angle()
    
            # Clear the screen
            self.screen.fill(self.BLACK)
    
            self.draw_radar_circle()
            self.draw_radar_sweep_line()
            self.draw_tracking_radar()
            self.draw_track_sweep_line()
    
            detection = self.check_object_detection()
            if detection:
                self.search_detections.append(detection)
    
            current_time = time.time()
            self.update_search_detections(current_time)
            self.update_track_history(current_time)
    
            self.calculate_tracking_data()
            self.draw_detections()
            self.draw_active_object()
            self.draw_tracking_information()
    
            # Draw velocity graph
            self.draw_velocity_graph()
            self.draw_predicted_path()
    
            for missile in self.missiles:
                missile.update()
                missile.draw(self.screen)

            # Remove destroyed missiles
            self.missiles = [missile for missile in self.missiles if missile.active]

            # Check if any missiles need to self-destruct
            if not self.missiles and self.missile and not self.missile.active:
                running = False  # End the simulation
            if self.current_angle >= 180 or self.current_angle <= 0:
                self.angle_increment = -self.angle_increment
    
            self.update_display()
            self.delay()
    
        # Quit Pygame
        pygame.quit()


# Create an instance of the Radar class
radar = Radar(
    width=1920,
    height=1080,
    radius=400,
    sweep_speed=2,
    angle_increment=1,
    track_angle_range=15,
    track_speed=1,
    track_sweep_speed=.5,
    max_detect_count=5,
    max_detect_age=5
)

# Run the radar simulation
radar.run()
