from FlannFeatureMatching import process_video
from edge_detector import EdgeDetector

# start some "UI"

# load a clean model
ed = EdgeDetector()
ed.load("emm.npy")

# wait for the first user to return the car
process_video("damaged.mp4", ed)

# wait for the second user to return the car
process_video("damaged2.mp4", ed)

# store if you want
ed.store("emm_damaged.npy")