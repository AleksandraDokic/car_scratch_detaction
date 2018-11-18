from FlannFeatureMatching import process_video
from edge_detector import EdgeDetector

# start some "UI"

# load a clean model
ed = EdgeDetector()

#process_video("original.mov", ed)
#ed.store("emm.npy")

ed.load("emm.npy")

# wait for the first user to return the car
process_video("damaged.mov", ed)
ed.store("emm_damaged1.npy")

# wait for the second user to return the car
ed = EdgeDetector()
ed.load("emm_damaged1.npy")
process_video("damaged2.mov", ed)
ed.store("emm_damaged2.npy")
#'''