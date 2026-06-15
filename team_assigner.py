from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:image.shape[0] // 2, :]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(top_half.reshape(-1, 3))

        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        background_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - background_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = [self.get_player_color(frame, p["bbox"]) for p in player_detections.values()]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
