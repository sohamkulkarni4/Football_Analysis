from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i:i + batch_size], conf=0.1)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            # Treat goalkeepers as players so they get the same tracking logic
            for i, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for det in detection_with_tracks:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for det in detection_sv:
                bbox = det[0].tolist()
                cls_id = det[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2),
                    axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235,
                    color=color, thickness=2, lineType=cv2.LINE_4)

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1 = x_center - rect_w // 2
            x2 = x_center + rect_w // 2
            y1 = y2 - rect_h // 2 + 15
            y2_r = y2 + rect_h // 2 + 15
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2_r)), color, cv2.FILLED)
            x_text = int(x1) + 12 - (10 if track_id > 99 else 0)
            cv2.putText(frame, f"{track_id}", (x_text, int(y1 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        frames_so_far = team_ball_control[:frame_num + 1]
        t1 = frames_so_far[frames_so_far == 1].shape[0]
        t2 = frames_so_far[frames_so_far == 2].shape[0]
        total = t1 + t2

        cv2.putText(frame, f"Team 1 Ball Control: {t1/total*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {t2/total*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in tracks["referees"][frame_num].items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in tracks["ball"][frame_num].items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames
