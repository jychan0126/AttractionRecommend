# Data
- **ai.user_type_prob.pkl** \
Use the type of attractions where user have been to to compute probability.
- **ai.user_type_smax.pkl** \
Use above result to through a softmax function.
- **type_attraction_dic.pkl** \
Format : `{"type" : [attr_1, attr_2, ...]}`
- **type_index.pkl** \
Format : `[type1, type2, ...]`
- **attractionToindex.pkl** \
Format : `[attr1, attr2, ...]`
- **region_attraction_dic.pkl** \
Format : `{"region" : [attr1, attr2, ...]}`

# How to run the program
`python test.py --t [8,30,39,40,44,62,63,75,83,97,102,103] --r [1] --m ai_NeuMF_64_[64,32,16,8]_1560311620.h5` \
--t Can refer to "type.txt". \
--r Can refer to "region.txt". \
--m Which model to apply.
