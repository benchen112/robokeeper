import unittest
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


if __name__=='__main__':
    unittest.main()
