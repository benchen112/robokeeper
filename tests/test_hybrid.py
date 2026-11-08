import unittest
from math import hypot
import cv2
import numpy as np

from robokeeper import BallTracker, FloorEstimator, HybridBallSegmenter


class HybridTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)

    def scene(self, moving_x=None, radius=12):
        frame = np.full((200, 320), 90, np.uint8)
        # Strong stationary circle distractors.
        for center in ((230, 45), (265, 150)):
            cv2.circle(frame, center, 15, 230, -1)
            cv2.circle(frame, center, 6, 25, -1)
        if moving_x is not None:
            cv2.circle(frame, (moving_x, 100), radius, 230, -1)
            cv2.circle(frame, (moving_x, 100), max(2,radius//3), 25, -1)
        return frame

    def tracker(self):
        return BallTracker(HybridBallSegmenter(auto_floor=False, verify_appearance=False), confirmation_hits=1,
                           adaptive_gate=True)

    def test_tiled_fixed_camera_motion_matches_whole_frame_motion(self):
        # The fixed-camera mask skips still tiles; it must equal the original
        # whole-frame computation pixel for pixel, including the >20% reset.
        rng = np.random.default_rng(3)
        tiled = HybridBallSegmenter(fixed_camera=True)
        whole = HybridBallSegmenter(fixed_camera=True)
        base = cv2.GaussianBlur(rng.integers(40, 200, (200, 322), dtype=np.uint8), (9, 9), 0)
        for i in range(40):
            frame = base.copy()
            if i == 25:
                frame = 255 - frame  # Scene change.
            frame = cv2.add(frame, 8 if i > 30 else 0)  # Lighting step.
            cv2.circle(frame, (20 + 7 * i, 120 - i), 9, 230, -1)
            noise = rng.normal(0, 3, frame.shape)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            self.assertEqual(tiled._motion(frame), whole._motion_general(frame))
            upsampled = cv2.resize(tiled.motion_mask, (322, 200), interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(upsampled, whole.motion_mask)
            for x0, y0, x1, y1 in ((0, 0, 322, 200), (3, 7, 101, 99), (40, 81, 77, 150)):
                self.assertEqual(tiled._mask_count(x0, y0, x1, y1), whole._mask_count(x0, y0, x1, y1))
        self.assertGreater(cv2.countNonZero(whole.motion_mask), 0)

    def test_stationary_round_distractors_do_not_acquire(self):
        tracker = self.tracker()
        for i in range(15):
            result = tracker.process(self.scene(), i/60)
            self.assertIsNone(result.center)

    def test_moving_ball_acquires_among_static_circles_at_multiple_sizes(self):
        for radius in (6, 12, 25):
            with self.subTest(radius=radius):
                tracker = self.tracker()
                observed = []
                for i in range(24):
                    x=30+i*4
                    result=tracker.process(self.scene(x,radius),i/60)
                    if result.observed:
                        observed.append(result)
                        self.assertLess(abs(result.center[0]-x),6)
                        self.assertLess(abs(result.center[1]-100),6)
                self.assertGreater(len(observed),10)

    def test_ball_first_seen_low_is_followed_above_the_acquisition_row(self):
        # A chip: the ball rises past acquisition_min_row before it is accepted.
        def run(min_row):
            tracker = BallTracker(HybridBallSegmenter(auto_floor=False, verify_appearance=False,
                                                      acquisition_min_row=min_row),
                                  confirmation_hits=1, adaptive_gate=True)
            rows = []
            for i in range(24):
                frame = self.scene()
                y = 150 - 6 * i
                cv2.circle(frame, (60 + i, y), 10, 230, -1)
                cv2.circle(frame, (60 + i, y), 3, 25, -1)
                result = tracker.process(frame, i / 60)
                if result.observed:
                    rows.append(result.center[1])
            return rows
        rows = run(min_row=140)
        self.assertTrue(rows and min(rows) < 100, rows)
        # A ball appearing high, never seen below the row, is still ignored.
        tracker = BallTracker(HybridBallSegmenter(auto_floor=False, verify_appearance=False,
                                                  acquisition_min_row=140),
                              confirmation_hits=1, adaptive_gate=True)
        for i in range(24):
            frame = self.scene()
            cv2.circle(frame, (60 + 4 * i, 60), 10, 230, -1)
            cv2.circle(frame, (60 + 4 * i, 60), 3, 25, -1)
            self.assertFalse(tracker.process(frame, i / 60).observed)

    def test_blob_proposals_fit_the_ball_edge(self):
        segmenter = HybridBallSegmenter(auto_floor=False, verify_appearance=False)
        frame = np.full((200, 320), 90, np.uint8)
        cv2.circle(frame, (150, 110), 20, 140, -1)
        segmenter._gray = cv2.GaussianBlur(frame, (3, 3), 0)
        # A motion blob a little off the ball (only part of it changed).
        proposals = segmenter._blob_proposals([(134, 92, 30, 34)])
        self.assertTrue(proposals)
        x, y, r = proposals[0]
        self.assertLess(hypot(x - 150, y - 110), 2)
        self.assertLess(abs(r - 20), 2)
        # Elongated blobs (a leg, a trail) propose nothing.
        self.assertEqual(segmenter._blob_proposals([(100, 50, 20, 90)]), [])

    def test_global_translation_does_not_turn_stationary_objects_into_tracks(self):
        rng=np.random.default_rng(123)
        base=self.scene()
        # Fixed features support robust scene alignment.
        for x,y in rng.integers((10,10),(300,185),size=(80,2)):
            cv2.rectangle(base,(int(x),int(y)),(int(x)+3,int(y)+3),150,-1)
        tracker=self.tracker()
        for i in range(14):
            matrix=np.float32([[1,0,i*.6],[0,1,i*.4]])
            frame=cv2.warpAffine(base,matrix,(320,200),borderMode=cv2.BORDER_REPLICATE)
            result=tracker.process(frame,i/60)
            self.assertIsNone(result.center)

    def test_locked_track_skips_hough_until_the_ball_is_lost(self):
        for growth in (0, 1):  # rolling across, then approaching the camera
            with self.subTest(growth=growth):
                tracker = self.tracker()
                tracker.segmenter.locked_hough_interval = 0
                hough = cv2.HoughCircles
                calls = []
                def counted(*args, **kwargs):
                    calls.append(frame)
                    return hough(*args, **kwargs)
                cv2.HoughCircles = counted
                try:
                    template_only = []
                    for frame in range(24):
                        x = 30+frame*4
                        result = tracker.process(self.scene(x, 12+growth*frame), frame/60)
                        if tracker.segmenter.template_only:
                            template_only.append(frame)
                            self.assertTrue(result.observed)
                            self.assertLess(abs(result.center[0]-x), 6)
                            self.assertLess(abs(result.center[1]-100), 6)
                    self.assertGreater(len(template_only), 10)
                    self.assertFalse(set(calls) & set(template_only))
                    calls.clear()
                    for frame in range(24, 28):
                        tracker.process(self.scene(), frame/60)
                    self.assertFalse(tracker.segmenter.template_only)
                    self.assertTrue(calls)
                finally:
                    cv2.HoughCircles = hough

    def test_recent_identity_rejects_size_jump_but_expires(self):
        tracker = self.tracker()
        for i in range(20):
            result = tracker.process(self.scene(30+i*4, 25), i/60)
        self.assertTrue(result.observed)
        for i in range(8):
            tracker.process(self.scene(), (20+i)/60)
        for i in range(10):
            result = tracker.process(self.scene(140+i*4, 6), (28+i)/60)
            self.assertFalse(result.observed)
        # A new shot can acquire a different apparent size after identity expires.
        observations = []
        for i in range(20):
            result = tracker.process(self.scene(30+i*4, 6), 2.5+i/60)
            observations.append(result.observed)
        self.assertGreater(sum(observations), 10)

    def test_uniform_scene_has_no_invented_floor(self):
        estimator=FloorEstimator()
        for _ in range(25):
            result=estimator.update(np.full((200,320),90,np.uint8))
        self.assertIsNone(result.boundary_y)
        self.assertEqual(result.confidence,0)

    def test_floor_estimate_tracks_a_supported_boundary(self):
        estimator=FloorEstimator()
        rng=np.random.default_rng(9)
        frame=np.full((200,320),90,np.uint8)
        frame[:95]=rng.integers(30,210,size=(95,320),dtype=np.uint8)
        cv2.line(frame,(0,95),(319,95),220,2)
        for _ in range(25):
            estimate=estimator.update(frame)
        self.assertIsNotNone(estimate.boundary_y)
        self.assertLess(abs(estimate.boundary_y-.475),.04)


    def test_floor_search_slows_once_found_and_pauses_while_tracking(self):
        rng=np.random.default_rng(9)
        frame=np.full((200,320),90,np.uint8)
        frame[:95]=rng.integers(30,210,size=(95,320),dtype=np.uint8)
        cv2.line(frame,(0,95),(319,95),220,2)
        lines=cv2.HoughLinesP
        searches=[]
        def counted(*args,**kwargs):
            searches.append(1)
            return lines(*args,**kwargs)
        cv2.HoughLinesP=counted
        try:
            estimator=FloorEstimator()
            for _ in range(300):
                estimator.update(frame)
            # Every 6th frame until three searches agree, then every 120th.
            self.assertEqual(len(searches),5)
            self.assertIsNotNone(estimator.estimate.boundary_y)
            searches.clear()
            segmenter=HybridBallSegmenter(verify_appearance=False)
            segmenter.set_tracking_hint((160.,100.),10.,30.)
            for _ in range(30):
                segmenter.find_candidates(frame)
            self.assertEqual(searches,[])
        finally:
            cv2.HoughLinesP=lines

    def test_large_ball_predicted_at_the_image_edge_does_not_crash(self):
        # fast_shortexp_R 2026-11-06: a 1-2 px search sliver at the border
        # shrank to zero pixels and cv2.resize raised.
        segmenter=HybridBallSegmenter(verify_appearance=False)
        frame=self.scene(300,25)
        segmenter._gray=frame
        segmenter._template=np.full((32,32),200,np.float32)
        for x in (frame.shape[1]+128.5, -128.5):
            segmenter.set_tracking_hint((x,100.),100.,30.)
            self.assertEqual(segmenter._template_proposals(frame),[])

if __name__=='__main__':
    unittest.main()
