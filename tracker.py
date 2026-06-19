"""
SORT Tracker (Simple Online and Realtime Tracking)
Implémentation légère sans dépendance scipy/filterpy.
Basé sur : Bewley et al. 2016 — arxiv.org/abs/1602.00763
"""
import numpy as np


class KalmanBoxTracker:
    """
    Suit un seul objet avec un filtre de Kalman.
    État : [x, y, s, r, dx, dy, ds]
      x,y = centre, s = surface, r = aspect ratio, dx/dy/ds = vélocités
    """
    count = 0

    def __init__(self, bbox):
        # Matrices du filtre de Kalman
        self.kf_F = np.eye(7, 7)   # transition d'état
        for i in range(4):
            self.kf_F[i, i + 3] = 1.0

        self.kf_H = np.eye(4, 7)   # mesure

        self.kf_R = np.eye(4) * 10      # bruit de mesure
        self.kf_R[2:, 2:] *= 10

        self.kf_P = np.eye(7) * 10     # covariance initiale
        self.kf_P[4:, 4:] *= 1000

        self.kf_Q = np.eye(7)          # bruit de processus
        self.kf_Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        z = self._bbox_to_z(bbox)
        self.x[:4] = z

        KalmanBoxTracker.count += 1
        self.id          = KalmanBoxTracker.count
        self.history     = []
        self.hits        = 0
        self.hit_streak  = 0
        self.age         = 0
        self.time_since  = 0

    def _bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h else 1.0
        return np.array([[x], [y], [s], [r]])

    def _x_to_bbox(self):
        s = self.x[2, 0]
        r = self.x[3, 0]
        if r <= 0 or s <= 0:
            return [0, 0, 1, 1]
        w = np.sqrt(s * r)
        h = s / w
        return [
            self.x[0, 0] - w / 2,
            self.x[1, 0] - h / 2,
            self.x[0, 0] + w / 2,
            self.x[1, 0] + h / 2,
        ]

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0
        # Prédiction
        self.x = self.kf_F @ self.x
        # Covariance
        self.P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q
        self.age += 1
        if self.time_since > 0:
            self.hit_streak = 0
        self.time_since += 1
        self.history.append(self._x_to_bbox())
        return self.history[-1]

    def update(self, bbox):
        self.time_since = 0
        self.hits += 1
        self.hit_streak += 1
        z = self._bbox_to_z(bbox)
        # Innovation
        y = z - self.kf_H @ self.x
        S = self.kf_H @ self.P @ self.kf_H.T + self.kf_R
        K = self.P @ self.kf_H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.kf_P = (np.eye(7) - K @ self.kf_H) @ self.P
        self.history = []

    @property
    def P(self):
        return self.kf_P

    @P.setter
    def P(self, v):
        self.kf_P = v

    def get_state(self):
        return self._x_to_bbox()


def iou_batch(bb_det, bb_trk):
    """IoU vectorisé entre N détections et M trackers."""
    bb_det = np.expand_dims(bb_det, 1)   # (N,1,4)
    bb_trk = np.expand_dims(bb_trk, 0)   # (1,M,4)

    xx1 = np.maximum(bb_det[:, :, 0], bb_trk[:, :, 0])
    yy1 = np.maximum(bb_det[:, :, 1], bb_trk[:, :, 1])
    xx2 = np.minimum(bb_det[:, :, 2], bb_trk[:, :, 2])
    yy2 = np.minimum(bb_det[:, :, 3], bb_trk[:, :, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    det_area = (bb_det[:, :, 2] - bb_det[:, :, 0]) * (bb_det[:, :, 3] - bb_det[:, :, 1])
    trk_area = (bb_trk[:, :, 2] - bb_trk[:, :, 0]) * (bb_trk[:, :, 3] - bb_trk[:, :, 1])
    union = det_area + trk_area - inter

    return inter / np.where(union == 0, 1e-9, union)


def linear_assignment(cost_matrix):
    """Algorithme hongrois simple (greedy fallback si scipy absent)."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost_matrix)
        return np.stack([r, c], axis=1)
    except ImportError:
        # greedy
        matches = []
        used_r, used_c = set(), set()
        idx = np.dstack(np.unravel_index(np.argsort(cost_matrix.ravel()), cost_matrix.shape))[0]
        for r, c in idx:
            if r not in used_r and c not in used_c:
                matches.append([r, c])
                used_r.add(r)
                used_c.add(c)
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))

    unmatched_dets = [d for d in range(len(detections))
                      if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers))
                      if t not in matched_indices[:, 1]]

    matches = [m for m in matched_indices
               if iou_matrix[m[0], m[1]] >= iou_threshold]
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age      = max_age
        self.min_hits     = min_hits
        self.iou_threshold = iou_threshold
        self.trackers     = []
        self.frame_count  = 0
        KalmanBoxTracker.count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets : np.array de shape (N,5) = [x1,y1,x2,y2,score]
        Retourne np.array (M,5) = [x1,y1,x2,y2,id]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since < 1 and (trk.hit_streak >= self.min_hits or
                                        self.frame_count <= self.min_hits):
                ret.append([*d, trk.id])
            i -= 1
            if trk.time_since > self.max_age:
                self.trackers.pop(i)

        return np.array(ret) if ret else np.empty((0, 5))
